// Capacity selected-row scan and analytic temporal credit. No Python token loop.
#include <torch/extension.h>
#include <cmath>
#include <limits>
#include <vector>

using at::Tensor;

template <typename T> inline T operand(T x, int64_t mode) {
  if(mode==1) return static_cast<T>(static_cast<at::Half>(x));
  if(mode==2) return static_cast<T>(static_cast<at::BFloat16>(x));
  return x;
}

Tensor operands(Tensor x,int64_t mode) {
  if(mode==0) return x;
  return x.to(mode==1?at::kHalf:at::kBFloat16).to(x.scalar_type());
}

struct RowTable {
  const void* data;
  at::ScalarType dtype;
  explicit RowTable(const Tensor& t):data(t.const_data_ptr()),dtype(t.scalar_type()) {
    TORCH_CHECK(t.device().is_cpu() && t.is_contiguous(), "Capacity row tables must be contiguous CPU tensors");
    TORCH_CHECK(dtype==at::kFloat || dtype==at::kDouble || dtype==at::kHalf || dtype==at::kBFloat16,
                "Capacity row tables must be floating point");
  }
  template<typename T> inline T get(int64_t index) const {
    switch(dtype) {
      case at::kFloat:return static_cast<T>(static_cast<const float*>(data)[index]);
      case at::kDouble:return static_cast<T>(static_cast<const double*>(data)[index]);
      case at::kHalf:return static_cast<T>(static_cast<const at::Half*>(data)[index]);
      case at::kBFloat16:return static_cast<T>(static_cast<const at::BFloat16*>(data)[index]);
      default:TORCH_CHECK(false,"Unsupported Capacity row dtype");
    }
  }
};

void same_storage_contract(const Tensor& x, const std::vector<Tensor>& tensors) {
  TORCH_CHECK(x.device().is_cpu() && x.is_contiguous(), "Capacity requires contiguous CPU tensors");
  for (const auto& t : tensors) {
    TORCH_CHECK(t.device().is_cpu() && t.is_contiguous() && t.scalar_type()==x.scalar_type(),
                "Capacity native floating inputs must share CPU device, dtype and contiguous layout");
  }
}

void check_indices(const Tensor& indices) {
  TORCH_CHECK(indices.device().is_cpu() && indices.is_contiguous() && indices.scalar_type()==at::kLong,
              "Capacity native indices must be contiguous CPU int64");
}

Tensor address(Tensor x, Tensor projection, int64_t active) {
  same_storage_contract(x,{projection});
  TORCH_CHECK(x.dim()==3 && (projection.dim()==2 || projection.dim()==3) && active>0);
  TORCH_CHECK(projection.size(-1)==x.size(2) && projection.size(-2)>0);
  const auto B=x.size(0), T=x.size(1), D=x.size(2);
  const auto L=projection.dim()==3 ? projection.size(0) : 1;
  const auto N=projection.size(projection.dim()-2);
  auto out=at::empty({B,T,active}, x.options().dtype(at::kLong));
  auto ids=out.data_ptr<int64_t>();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "capacity_address", [&] {
    auto xp=x.data_ptr<scalar_t>(), pp=projection.data_ptr<scalar_t>();
    for(int64_t b=0;b<B;++b) for(int64_t t=0;t<T;++t) {
      int64_t bucket=0;
      for(int64_t l=0;l<L;++l) {
        scalar_t best=-std::numeric_limits<scalar_t>::infinity(); int64_t digit=0;
        for(int64_t n=0;n<N;++n) {
          scalar_t score=0;
          for(int64_t d=0;d<D;++d) score+=xp[(b*T+t)*D+d]*pp[(l*N+n)*D+d];
          // Strict comparison deliberately retains the first row on a tie.
          if(score>best || (std::isnan(score) && !std::isnan(best))) {best=score;digit=n;}
        }
        bucket=bucket*N+digit;
      }
      for(int64_t a=0;a<active;++a) ids[(b*T+t)*active+a]=bucket*active+a;
    }
  });
  return out;
}

std::vector<Tensor> forward(Tensor x, Tensor key, Tensor bias, Tensor logit,
                           Tensor value, Tensor gain, Tensor features,
                           Tensor indices, bool sparse, int64_t operand_mode) {
  same_storage_contract(x,{features}); check_indices(indices);
  if(!sparse) same_storage_contract(x,{key,bias,logit,value,gain});
  RowTable keys(key), biases(bias), logits(logit), values(value), gains(gain);
  TORCH_CHECK(x.dim()==3 && key.dim()==2 && value.dim()==2 && gain.numel()==1);
  const auto B=x.size(0), T=x.size(1), D=x.size(2), C=key.size(0), S=value.size(1);
  TORCH_CHECK(key.size(1)==D && bias.sizes()==at::IntArrayRef({C}) && logit.sizes()==bias.sizes());
  TORCH_CHECK(value.size(0)==C && features.sizes()==at::IntArrayRef({B,C}));
  TORCH_CHECK(!sparse || (indices.dim()==3 && indices.size(0)==B && indices.size(1)==T));
  const auto A=sparse ? indices.size(2) : C;
  auto out=at::zeros({B,T,S},x.options());
  auto acts=at::empty({B,T,A},x.options()), old=at::empty_like(acts), states=at::empty_like(acts);
  if(!sparse) {
    // Dense projections and output contraction are BLAS work; only the
    // recurrent dependency belongs in the scalar temporal scan.
    acts=operands(at::tanh(operands(at::matmul(x.reshape({B*T,D}),
         operands(key,operand_mode).transpose(0,1))+operands(bias,operand_mode),operand_mode)),operand_mode).reshape({B,T,C});
    auto rho=at::sigmoid(logit);
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "capacity_dense_recurrence", [&] {
      auto ap=acts.data_ptr<scalar_t>(),rp=rho.data_ptr<scalar_t>(),zp=features.data_ptr<scalar_t>();
      auto op=old.data_ptr<scalar_t>(),sp=states.data_ptr<scalar_t>();
      for(int64_t b=0;b<B;++b) for(int64_t t=0;t<T;++t) for(int64_t c=0;c<C;++c) {
        auto q=(b*T+t)*C+c;
        scalar_t prev=zp[b*C+c], next=rp[c]*prev+(1-rp[c])*ap[q];
        op[q]=prev;sp[q]=next;zp[b*C+c]=next;
      }
    });
    out=(at::matmul(operands(states.reshape({B*T,C}),operand_mode),
                   operands(value,operand_mode))*gain).reshape({B,T,S});
    return {out,acts,old,states};
  }
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "capacity_forward", [&] {
    auto xp=x.data_ptr<scalar_t>(), zp=features.data_ptr<scalar_t>();
    auto yp=out.data_ptr<scalar_t>(), ap=acts.data_ptr<scalar_t>(), op=old.data_ptr<scalar_t>(), sp=states.data_ptr<scalar_t>();
    auto ip=indices.data_ptr<int64_t>(); const scalar_t g=gain.item<scalar_t>();
    for(int64_t b=0;b<B;++b) for(int64_t t=0;t<T;++t) for(int64_t a=0;a<A;++a) {
      auto q=(b*T+t)*A+a, r=sparse ? ip[q] : a;
      TORCH_CHECK(r>=0 && r<C,"Capacity row index out of range");
      scalar_t dot=operand(biases.get<scalar_t>(r),operand_mode);
      for(int64_t d=0;d<D;++d) dot+=xp[(b*T+t)*D+d]*operand(keys.get<scalar_t>(r*D+d),operand_mode);
      scalar_t act=operand(std::tanh(operand(dot,operand_mode)),operand_mode), rho=1/(1+std::exp(-logits.get<scalar_t>(r)));
      scalar_t prev=zp[b*C+r], next=rho*prev+(1-rho)*act;
      ap[q]=act; op[q]=prev; sp[q]=next; zp[b*C+r]=next;
      for(int64_t s=0;s<S;++s) yp[(b*T+t)*S+s]+=g*operand(next,operand_mode)*operand(values.get<scalar_t>(r*S+s),operand_mode);
    }
  });
  return {out,acts,old,states};
}

std::vector<Tensor> backward(Tensor x, Tensor key, Tensor logit, Tensor value, Tensor gain,
    Tensor indices, Tensor local, int64_t rows, Tensor acts, Tensor old, Tensor states,
    Tensor gout, Tensor gfinal, Tensor ek, Tensor eb, Tensor er, bool sparse, bool online,
    int64_t operand_mode) {
  same_storage_contract(x,{acts,old,states,gout,gfinal,ek,eb,er});
  if(!sparse) same_storage_contract(x,{key,logit,value,gain});
  RowTable keys(key),logits(logit),values(value),gains(gain);
  check_indices(indices); check_indices(local);
  TORCH_CHECK(x.dim()==3 && key.dim()==2 && value.dim()==2 && acts.dim()==3 && gain.numel()==1 && rows>=0);
  const auto B=x.size(0),T=x.size(1),D=x.size(2),C=key.size(0),S=value.size(1),A=acts.size(2);
  TORCH_CHECK(key.size(1)==D && value.size(0)==C && logit.sizes()==at::IntArrayRef({C}));
  TORCH_CHECK(acts.size(0)==B && acts.size(1)==T && old.sizes()==acts.sizes() && states.sizes()==acts.sizes());
  TORCH_CHECK(gout.sizes()==at::IntArrayRef({B,T,S}));
  TORCH_CHECK(!sparse || (indices.sizes()==acts.sizes() && local.sizes()==indices.sizes()));
  TORCH_CHECK(sparse || A==C);
  if(online) {
    TORCH_CHECK(ek.sizes()==at::IntArrayRef({B,C,D}) && eb.sizes()==at::IntArrayRef({B,C}) && er.sizes()==eb.sizes());
  } else { TORCH_CHECK(gfinal.sizes()==at::IntArrayRef({B,C})); }
  if(!sparse) {
    auto rho=at::sigmoid(logit);
    auto direct=at::matmul(gout.reshape({B*T,S}),operands(value,operand_mode).transpose(0,1)).reshape({B,T,C});
    auto used_states=operands(states,operand_mode);
    auto gv=at::matmul(used_states.reshape({B*T,C}).transpose(0,1),gout.reshape({B*T,S}))*gain;
    auto gg=(direct*used_states).sum();
    auto adjoint=online?at::zeros({B,C},x.options()):gfinal.clone();
    auto da=at::empty_like(acts), gr=at::zeros({C},x.options());
    auto input_da=online?at::empty_like(acts):da;
    auto eligibility_weight=online?at::empty_like(acts):at::empty({0},x.options());
    auto retention_weight=online?at::zeros({B,C},x.options()):at::empty({0},x.options());
    auto decay=at::ones({C},x.options());
    AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "capacity_dense_reverse", [&] {
      auto rp=rho.data_ptr<scalar_t>(),dp=direct.data_ptr<scalar_t>(),ap=acts.data_ptr<scalar_t>();
      auto op=old.data_ptr<scalar_t>(),dz=adjoint.data_ptr<scalar_t>(),dap=da.data_ptr<scalar_t>();
      auto idap=input_da.data_ptr<scalar_t>(),grp=gr.data_ptr<scalar_t>();
      auto wp=eligibility_weight.data_ptr<scalar_t>(),rw=retention_weight.data_ptr<scalar_t>();
      auto cp=decay.data_ptr<scalar_t>(); scalar_t g=gain.item<scalar_t>();
      for(int64_t c=0;c<C;++c) {
        for(int64_t b=0;b<B;++b) {
          scalar_t power=1;
          for(int64_t ti=0;ti<T;++ti) {
            auto t=T-1-ti,q=(b*T+t)*C+c;
            scalar_t slope=(1-rp[c])*(1-ap[q]*ap[q]);
            scalar_t slope_rho=rp[c]*(1-rp[c])*(op[q]-ap[q]);
            scalar_t direct_grad=g*dp[q], full_grad=direct_grad+dz[b*C+c];
            dap[q]=full_grad*slope;
            grp[c]+=full_grad*slope_rho;
            dz[b*C+c]=full_grad*rp[c];
            if(online) {
              idap[q]=direct_grad*slope;
              wp[q]=power*slope;
              rw[b*C+c]+=power*slope_rho;
              power*=rp[c];
            }
          }
          if(online) cp[c]=power;
        }
      }
    });
    auto gk=at::matmul(da.reshape({B*T,C}).transpose(0,1),x.reshape({B*T,D}));
    auto gb=da.sum(at::IntArrayRef({0,1}));
    auto gx=at::matmul(input_da.reshape({B*T,C}),operands(key,operand_mode)).reshape({B,T,D});
    if(online) {
      // Sum v4's forward eligibility recurrence analytically. Incoming traces
      // contribute via the initial-state adjoint. Input VJP above stays local.
      gk+=(adjoint.unsqueeze(-1)*ek).sum(0);
      gb+=(adjoint*eb).sum(0);
      gr+=(adjoint*er).sum(0);
      ek.mul_(decay.reshape({1,C,1})).add_(at::bmm(eligibility_weight.transpose(1,2),x));
      eb.mul_(decay).add_(eligibility_weight.sum(1));
      er.mul_(decay).add_(retention_weight);
      return {gx,gk,gb,gr,gv,gg,at::empty({0},x.options())};
    }
    return {gx,gk,gb,gr,gv,gg,adjoint};
  }
  auto gx=at::zeros_like(x), gk=at::zeros({rows,D},x.options()), gb=at::zeros({rows},x.options());
  auto gr=at::zeros_like(gb), gv=at::zeros({rows,S},x.options()), gg=at::zeros({},x.options());
  auto gp=online ? at::empty({0},x.options()) : gfinal.clone();
  AT_DISPATCH_FLOATING_TYPES(x.scalar_type(), "capacity_backward", [&] {
    auto xp=x.data_ptr<scalar_t>();
    auto ap=acts.data_ptr<scalar_t>(),op=old.data_ptr<scalar_t>(),sp=states.data_ptr<scalar_t>();
    auto dy=gout.data_ptr<scalar_t>(),dx=gx.data_ptr<scalar_t>(),dk=gk.data_ptr<scalar_t>();
    auto db=gb.data_ptr<scalar_t>(),dr=gr.data_ptr<scalar_t>(),dv=gv.data_ptr<scalar_t>(),dg=gg.data_ptr<scalar_t>();
    auto dz=gp.data_ptr<scalar_t>(), ekey=ek.data_ptr<scalar_t>(),ebias=eb.data_ptr<scalar_t>(),erho=er.data_ptr<scalar_t>();
    auto ip=indices.data_ptr<int64_t>(),lp=local.data_ptr<int64_t>();
    const scalar_t g=gain.item<scalar_t>();
    for(int64_t b=0;b<B;++b) for(int64_t ti=0;ti<T;++ti) {
      int64_t t=online?ti:T-1-ti;
      for(int64_t a=0;a<A;++a) {
        auto q=(b*T+t)*A+a,r=sparse?ip[q]:a,j=sparse?lp[q]:a;
        TORCH_CHECK(r>=0 && r<C && j>=0 && j<rows,"Capacity row index out of range");
        scalar_t rho=1/(1+std::exp(-logits.get<scalar_t>(r))),act=ap[q],slope=(1-rho)*(1-act*act),gz=0;
        for(int64_t s=0;s<S;++s) {
          scalar_t cot=dy[(b*T+t)*S+s];
          auto v=operand(values.get<scalar_t>(r*S+s),operand_mode), z=operand(sp[q],operand_mode);
          gz+=g*cot*v; dv[j*S+s]+=g*z*cot; *dg+=cot*z*v;
        }
        if(online) {
          auto br=b*C+r;
          ebias[br]=rho*ebias[br]+slope;
          erho[br]=rho*erho[br]+rho*(1-rho)*(op[q]-act);
          db[j]+=gz*ebias[br]; dr[j]+=gz*erho[br];
          for(int64_t d=0;d<D;++d) {
            auto k=br*D+d;
            ekey[k]=rho*ekey[k]+slope*xp[(b*T+t)*D+d];
            dk[j*D+d]+=gz*ekey[k];
            dx[(b*T+t)*D+d]+=gz*slope*operand(keys.get<scalar_t>(r*D+d),operand_mode);
          }
        } else {
          gz+=dz[b*C+r]; dz[b*C+r]=gz*rho;
          db[j]+=gz*slope; dr[j]+=gz*rho*(1-rho)*(op[q]-act);
          for(int64_t d=0;d<D;++d) {
            dk[j*D+d]+=gz*slope*xp[(b*T+t)*D+d];
            dx[(b*T+t)*D+d]+=gz*slope*operand(keys.get<scalar_t>(r*D+d),operand_mode);
          }
        }
      }
    }
  });
  return {gx,gk,gb,gr,gv,gg,gp};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m) {
  m.def("address",&address); m.def("forward",&forward); m.def("backward",&backward);
}
