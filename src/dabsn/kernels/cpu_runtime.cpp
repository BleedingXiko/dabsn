// Native C++/OpenMP fused DABSN core scan for CPU framework execution.
//
// This implements the same recurrence as the Triton and PyTorch reference core
// scans, with the time loop, recurrent state, and pointwise math in C++. The
// per-step recurrent matrix-vector product uses a cache-friendly row loop.
// OpenMP parallelizes over the batch dimension.
//
// Built JIT by cpu.py via torch.utils.cpp_extension. Math must match the
// reference exactly.

#include <torch/extension.h>
#include <vector>
#include <array>
#include <cmath>
#include <algorithm>
#include <limits>
#include <string>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace {

inline float sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float softplusf(float x) {
  // numerically stable log1p(exp(x)) == max(x,0)+log1p(exp(-|x|))
  return std::max(x, 0.0f) + std::log1p(std::exp(-std::fabs(x)));
}
inline float hardsigmoidf(float x) {
  // torch F.hardsigmoid == relu6(x+3)/6 == clamp(x+3,0,6)/6
  return std::min(std::max(x + 3.0f, 0.0f), 6.0f) * (1.0f / 6.0f);
}

struct OnlineDual {
  float value = 0.0f;
  std::array<float, 23> derivative{};
};

inline OnlineDual online_constant(float value) {
  OnlineDual result;
  result.value = value;
  return result;
}

inline OnlineDual online_basis(float value, int index) {
  OnlineDual result = online_constant(value);
  result.derivative[index] = 1.0f;
  return result;
}

inline OnlineDual online_add(const OnlineDual& lhs, const OnlineDual& rhs) {
  OnlineDual result;
  result.value = lhs.value + rhs.value;
  for (int i = 0; i < 23; ++i) result.derivative[i] = lhs.derivative[i] + rhs.derivative[i];
  return result;
}

inline OnlineDual online_sub(const OnlineDual& lhs, const OnlineDual& rhs) {
  OnlineDual result;
  result.value = lhs.value - rhs.value;
  for (int i = 0; i < 23; ++i) result.derivative[i] = lhs.derivative[i] - rhs.derivative[i];
  return result;
}

inline OnlineDual online_mul(const OnlineDual& lhs, const OnlineDual& rhs) {
  OnlineDual result;
  result.value = lhs.value * rhs.value;
  for (int i = 0; i < 23; ++i) {
    result.derivative[i] = lhs.derivative[i] * rhs.value + lhs.value * rhs.derivative[i];
  }
  return result;
}

inline OnlineDual online_tanh(const OnlineDual& input) {
  OnlineDual result;
  result.value = std::tanh(input.value);
  const float scale = 1.0f - result.value * result.value;
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * scale;
  return result;
}

inline OnlineDual online_abs(const OnlineDual& input) {
  OnlineDual result;
  result.value = std::fabs(input.value);
  const float scale = input.value > 0.0f ? 1.0f : (input.value < 0.0f ? -1.0f : 0.0f);
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * scale;
  return result;
}

inline OnlineDual online_exp(const OnlineDual& input) {
  OnlineDual result;
  result.value = std::exp(input.value);
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * result.value;
  return result;
}

inline OnlineDual online_sigmoid(const OnlineDual& input) {
  OnlineDual result;
  result.value = sigmoidf(input.value);
  const float scale = result.value * (1.0f - result.value);
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * scale;
  return result;
}

inline OnlineDual online_softplus(const OnlineDual& input) {
  OnlineDual result;
  result.value = softplusf(input.value);
  const float scale = sigmoidf(input.value);
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * scale;
  return result;
}

inline OnlineDual online_hardsigmoid(const OnlineDual& input) {
  OnlineDual result;
  result.value = hardsigmoidf(input.value);
  const float scale = input.value > -3.0f && input.value < 3.0f ? 1.0f / 6.0f : 0.0f;
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * scale;
  return result;
}

inline OnlineDual online_clamp01(const OnlineDual& input) {
  OnlineDual result;
  result.value = std::min(std::max(input.value, 0.0f), 1.0f);
  const float scale = input.value > 0.0f && input.value < 1.0f ? 1.0f : 0.0f;
  for (int i = 0; i < 23; ++i) result.derivative[i] = input.derivative[i] * scale;
  return result;
}

inline float norm_clamp(const std::vector<float>& x) {
  double ss = 0.0;
  for (float v : x) ss += (double)v * (double)v;
  return std::max((float)std::sqrt(ss), 1.0e-12f);
}

inline float dot_ptr_vec(const float* a, const std::vector<float>& b, int64_t n) {
  float s = 0.0f;
  for (int64_t i = 0; i < n; ++i) s += a[i] * b[i];
  return s;
}

inline double portable_exp64(double x) {
  constexpr double INV_LN2 = 1.4426950408889634;
  constexpr double LN2_HI = 0.6931471803691238;
  constexpr double LN2_LO = 1.9082149292705877e-10;
  constexpr double C[] = {
      1.0, 1.0, 0.5, 1.0 / 6.0, 1.0 / 24.0, 1.0 / 120.0,
      1.0 / 720.0, 1.0 / 5040.0, 1.0 / 40320.0, 1.0 / 362880.0,
      1.0 / 3628800.0, 1.0 / 39916800.0, 1.0 / 479001600.0,
      1.0 / 6227020800.0};
  const double k = std::nearbyint(x * INV_LN2);
  const double r = (x - k * LN2_HI) - k * LN2_LO;
  double acc = C[13];
  for (int i = 12; i >= 0; --i) acc = acc * r + C[i];
  return std::ldexp(acc, static_cast<int>(k));
}

inline double portable_tanh64(double x) {
  const double a = std::min(std::fabs(x), 40.0);
  const double mag = 1.0 - 2.0 / (portable_exp64(2.0 * a) + 1.0);
  return x > 0.0 ? mag : (x < 0.0 ? -mag : 0.0);
}

inline double portable_log64(double x) {
  int exponent = 0;
  double mantissa = std::frexp(std::max(x, std::numeric_limits<double>::min()), &exponent);
  if (mantissa < 2.0 / 3.0) {
    mantissa *= 2.0;
    --exponent;
  }
  const double t = (mantissa - 1.0) / (mantissa + 1.0);
  const double t2 = t * t;
  double acc = 1.0 / 19.0;
  for (int d = 17; d >= 1; d -= 2) acc = acc * t2 + 1.0 / static_cast<double>(d);
  return static_cast<double>(exponent) * 0.6931471805599453 + 2.0 * t * acc;
}

inline double portable_log1p64(double x) {
  const double u = 1.0 + x;
  if (u == 1.0) return x;
  return portable_log64(u) * (x / std::max(u - 1.0, std::numeric_limits<double>::min()));
}

inline bool use_local_field_blas_shape(int64_t B, int64_t T) {
  return B >= 256 && T <= 16;
}

template <typename scalar_t>
void linrec_forward_impl(
    const scalar_t* ap, const scalar_t* gp, const scalar_t* yp, scalar_t* op,
    int64_t B, int64_t T, int64_t H) {
  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> carry(H);
    for (int64_t h = 0; h < H; ++h) carry[h] = static_cast<float>(yp[bi * H + h]);
    for (int64_t t = 0; t < T; ++t) {
      const int64_t off = (bi * T + t) * H;
      for (int64_t h = 0; h < H; ++h) {
        carry[h] = static_cast<float>(ap[off + h]) * carry[h] + static_cast<float>(gp[off + h]);
        op[off + h] = static_cast<scalar_t>(carry[h]);
      }
    }
  }
}

template <typename scalar_t>
void linrec_backward_impl(
    const scalar_t* ap, const scalar_t* gp, const scalar_t* yp, const scalar_t* gyp,
    scalar_t* dap, scalar_t* dgp, scalar_t* dyp,
    int64_t B, int64_t T, int64_t H) {
  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> yprev(T * H);
    std::vector<float> carry_fwd(H);
    for (int64_t h = 0; h < H; ++h) carry_fwd[h] = static_cast<float>(yp[bi * H + h]);
    for (int64_t t = 0; t < T; ++t) {
      const int64_t off = (bi * T + t) * H;
      for (int64_t h = 0; h < H; ++h) {
        yprev[t * H + h] = carry_fwd[h];
        carry_fwd[h] = static_cast<float>(ap[off + h]) * carry_fwd[h] + static_cast<float>(gp[off + h]);
      }
    }
    std::vector<float> carry(H, 0.0f);
    for (int64_t tr = 0; tr < T; ++tr) {
      const int64_t t = T - 1 - tr;
      const int64_t off = (bi * T + t) * H;
      for (int64_t h = 0; h < H; ++h) {
        const float a_next = (t + 1 < T) ? static_cast<float>(ap[(bi * T + t + 1) * H + h]) : 0.0f;
        const float c = static_cast<float>(gyp[off + h]) + a_next * carry[h];
        dgp[off + h] = static_cast<scalar_t>(c);
        dap[off + h] = static_cast<scalar_t>(c * yprev[t * H + h]);
        carry[h] = c;
      }
    }
    for (int64_t h = 0; h < H; ++h) {
      dyp[bi * H + h] = static_cast<scalar_t>(static_cast<float>(ap[bi * T * H + h]) * carry[h]);
    }
  }
}

}  // namespace


std::vector<torch::Tensor> dabsn_core_scan_cpu(
    torch::Tensor Wx, torch::Tensor Wgx, torch::Tensor GA_w,   // GA_w=(2H,H): [Ug;A]
    torch::Tensor beta, torch::Tensor log_kappa, torch::Tensor logit_recover,
    torch::Tensor k_s, torch::Tensor k_y, torch::Tensor k_b, torch::Tensor k_n, torch::Tensor k_bias,
    torch::Tensor r_s, torch::Tensor r_y, torch::Tensor r_b, torch::Tensor r_n, torch::Tensor r_bias,
    torch::Tensor logit_c_decay, torch::Tensor k_c, torch::Tensor r_c,
    torch::Tensor initial_b, torch::Tensor initial_e, torch::Tensor initial_c,
    double gate_alpha_d, double lam_d, double c_suppress_d) {

  Wx = Wx.contiguous(); Wgx = Wgx.contiguous(); GA_w = GA_w.contiguous();
  initial_b = initial_b.contiguous(); initial_e = initial_e.contiguous(); initial_c = initial_c.contiguous();
  const int64_t B = Wx.size(0), T = Wx.size(1), H = Wx.size(2);

  auto opts = Wx.options().dtype(torch::kFloat32);
  torch::Tensor U     = torch::empty({B, T, 2 * H}, opts);
  torch::Tensor novel = torch::empty({B, T, H}, opts);
  torch::Tensor pout  = torch::empty({B, T, H}, opts);
  torch::Tensor ayout = torch::empty({B, T, H}, opts);
  torch::Tensor wout  = torch::empty({B, T, H}, opts);
  // cocktail tape (for the admission read): energy = PRE-update e, saturation = updated c.
  torch::Tensor eout  = torch::empty({B, T, H}, opts);
  torch::Tensor cout  = torch::empty({B, T, H}, opts);
  torch::Tensor bfinal = torch::empty({B, H}, opts);
  torch::Tensor efinal = torch::empty({B, H}, opts);
  torch::Tensor cfinal = torch::empty({B, H}, opts);

  const float* Wxp  = Wx.data_ptr<float>();
  const float* Wgxp = Wgx.data_ptr<float>();
  const float* GA   = GA_w.data_ptr<float>();      // row-major (2H, H)
  const float* betap = beta.data_ptr<float>();
  const float* logk  = log_kappa.data_ptr<float>();
  const float* logr  = logit_recover.data_ptr<float>();
  const float* ks = k_s.data_ptr<float>(); const float* ky = k_y.data_ptr<float>();
  const float* kb = k_b.data_ptr<float>(); const float* kn = k_n.data_ptr<float>();
  const float* kbias = k_bias.data_ptr<float>();
  const float* rs = r_s.data_ptr<float>(); const float* ry = r_y.data_ptr<float>();
  const float* rb = r_b.data_ptr<float>(); const float* rn = r_n.data_ptr<float>();
  const float* rbias = r_bias.data_ptr<float>();
  const float* lcd = logit_c_decay.data_ptr<float>();
  const float* kcp = k_c.data_ptr<float>(); const float* rcp = r_c.data_ptr<float>();
  const float* Binit = initial_b.data_ptr<float>();
  const float* Einit = initial_e.data_ptr<float>();
  const float* Cinit = initial_c.data_ptr<float>();

  float* Up = U.data_ptr<float>(); float* Np = novel.data_ptr<float>();
  float* Pp = pout.data_ptr<float>(); float* Ap = ayout.data_ptr<float>();
  float* Wp = wout.data_ptr<float>();
  float* Ep = eout.data_ptr<float>(); float* Cp = cout.data_ptr<float>();
  float* Bfinal = bfinal.data_ptr<float>();
  float* Efinal = efinal.data_ptr<float>();
  float* Cfinal = cfinal.data_ptr<float>();

  const float gate_alpha = (float)gate_alpha_d;
  const float lam = (float)lam_d;
  const float c_suppress = (float)c_suppress_d;

  if (use_local_field_blas_shape(B, T)) {
    torch::Tensor b = initial_b.clone();
    torch::Tensor e = initial_e.clone();
    torch::Tensor c = initial_c.clone();
    torch::Tensor GA_t = GA_w.transpose(0, 1).contiguous();
    torch::Tensor gate_alpha_t = torch::tensor(gate_alpha, opts);
    torch::Tensor lam_t = torch::tensor(lam, opts);
    torch::Tensor c_suppress_t = torch::tensor(c_suppress, opts);
    torch::Tensor kappa = torch::nn::functional::softplus(log_kappa);
    torch::Tensor recover = torch::sigmoid(logit_recover);
    torch::Tensor c_decay = torch::sigmoid(logit_c_decay);

    for (int64_t t = 0; t < T; ++t) {
      torch::Tensor wx = Wx.select(1, t);
      torch::Tensor wgx = Wgx.select(1, t);
      torch::Tensor y = torch::tanh(wx + b);
      torch::Tensor ugay = torch::matmul(y, GA_t);
      torch::Tensor ug = ugay.slice(1, 0, H);
      torch::Tensor ay = ugay.slice(1, H, 2 * H);
      torch::Tensor s = torch::clamp(wgx + ug + 3.0, 0.0, 6.0) * (1.0 / 6.0);
      torch::Tensor novelty = torch::tanh(torch::abs(ay - b));
      torch::Tensor stress = novelty * (1.0 - e);
      c = c_decay * c + (1.0 - c_decay) * stress;
      torch::Tensor novelty_eff = novelty * (1.0 - c_suppress_t * c);
      torch::Tensor energy_nov = novelty_eff;
      torch::Tensor tb = torch::tanh(b);
      torch::Tensor k_sig = k_s * s + k_y * y + k_b * tb + k_n * energy_nov + k_bias;
      torch::Tensor r_sig = r_s * s + r_y * y + r_b * tb + r_n * energy_nov + r_bias;
      k_sig = k_sig + k_c * c;
      r_sig = r_sig + r_c * c;
      torch::Tensor k_t = kappa * torch::exp(0.5 * torch::tanh(k_sig));
      torch::Tensor r_t = torch::clamp(recover * torch::exp(0.5 * torch::tanh(r_sig)), 0.0, 1.0);
      torch::Tensor p = s * e;
      torch::Tensor write = p * ay;
      torch::Tensor bn = (1.0 - gate_alpha_t) * b + beta + lam_t * write;
      torch::Tensor en = torch::clamp(e + r_t * (1.0 - e) - k_t * p, 0.0, 1.0);

      torch::Tensor Ut = U.select(1, t);
      Ut.slice(1, 0, H).copy_(y);
      Ut.slice(1, H, 2 * H).copy_(bn);
      novel.select(1, t).copy_(novelty);
      pout.select(1, t).copy_(p);
      ayout.select(1, t).copy_(ay);
      wout.select(1, t).copy_(write);
      eout.select(1, t).copy_(e);
      cout.select(1, t).copy_(c);
      b = bn;
      e = en;
    }
    return {U, novel, pout, ayout, wout, eout, cout, b, e, c};
  }

  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> b(H), e(H), c(H);
    for (int64_t i = 0; i < H; ++i) {
      b[i] = Binit[bi * H + i];
      e[i] = Einit[bi * H + i];
      c[i] = Cinit[bi * H + i];
    }
    std::vector<float> y(H), ug(H), ay(H);
    for (int64_t t = 0; t < T; ++t) {
      const float* wx  = Wxp  + (bi * T + t) * H;
      const float* wgx = Wgxp + (bi * T + t) * H;
      // y = tanh(Wx + b)
      for (int64_t i = 0; i < H; ++i) y[i] = std::tanh(wx[i] + b[i]);
      // [ug; ay] = GA_w @ y   (GA_w rows 0..H-1 = Ug, rows H..2H-1 = A)
      for (int64_t i = 0; i < H; ++i) {
        const float* gu = GA + i * H;
        const float* ga = GA + (H + i) * H;
        float su = 0.0f, sa = 0.0f;
        for (int64_t j = 0; j < H; ++j) { su += gu[j] * y[j]; sa += ga[j] * y[j]; }
        ug[i] = su; ay[i] = sa;
      }
      float* up = Up + (bi * T + t) * 2 * H;
      float* np = Np + (bi * T + t) * H;
      float* pp = Pp + (bi * T + t) * H;
      float* ap = Ap + (bi * T + t) * H;
      float* wp = Wp + (bi * T + t) * H;
      float* eg = Ep + (bi * T + t) * H;
      float* cg = Cp + (bi * T + t) * H;
      for (int64_t i = 0; i < H; ++i) {
        float s = hardsigmoidf(wgx[i] + ug[i]);
        float novelty = std::tanh(std::fabs(ay[i] - b[i]));
        float c_decay = sigmoidf(lcd[i]);
        float stress = novelty * (1.0f - e[i]);
        c[i] = c_decay * c[i] + (1.0f - c_decay) * stress;
        float novelty_eff = novelty * (1.0f - c_suppress * c[i]);
        float energy_nov = novelty_eff;
        float tb = std::tanh(b[i]);
        float k_sig = ks[i] * s + ky[i] * y[i] + kb[i] * tb + kn[i] * energy_nov + kbias[i];
        float r_sig = rs[i] * s + ry[i] * y[i] + rb[i] * tb + rn[i] * energy_nov + rbias[i];
        k_sig += kcp[i] * c[i]; r_sig += rcp[i] * c[i];
        float kappa = softplusf(logk[i]);
        float recover = sigmoidf(logr[i]);
        float k_t = kappa * std::exp(0.5f * std::tanh(k_sig));
        float r_t = std::min(std::max(recover * std::exp(0.5f * std::tanh(r_sig)), 0.0f), 1.0f);
        float p = s * e[i];                                 // uses OLD e
        float write = p * ay[i];
        float bn = (1.0f - gate_alpha) * b[i] + betap[i] + lam * (p * ay[i]);
        float en = std::min(std::max(e[i] + r_t * (1.0f - e[i]) - k_t * p, 0.0f), 1.0f);
        // outputs (U = cat[y, b_new]); cocktail: energy = OLD e (pre-update), saturation = c
        up[i] = y[i]; up[H + i] = bn;
        np[i] = novelty; pp[i] = p; ap[i] = ay[i]; wp[i] = write;
        eg[i] = e[i]; cg[i] = c[i];
        b[i] = bn; e[i] = en;
      }
    }
    for (int64_t i = 0; i < H; ++i) {
      Bfinal[bi * H + i] = b[i];
      Efinal[bi * H + i] = e[i];
      Cfinal[bi * H + i] = c[i];
    }
  }
  return {U, novel, pout, ayout, wout, eout, cout, bfinal, efinal, cfinal};
}

torch::Tensor three_way_read_cpu(
    torch::Tensor q, torch::Tensor kb, torch::Tensor wb, torch::Tensor wb_next,
    torch::Tensor cocktail, torch::Tensor cb,
    torch::Tensor key_bias_g, torch::Tensor adm_g,
    torch::Tensor allow, torch::Tensor induct_allow,
    torch::Tensor has_elig, torch::Tensor induct_elig,
    double scale_d, double cocktail_gain_d,
    double short_gain_d, double pad_gain_d, double induct_gain_d) {

  q = q.contiguous();
  kb = kb.contiguous();
  wb = wb.contiguous();
  wb_next = wb_next.contiguous();
  cocktail = cocktail.contiguous();
  cb = cb.contiguous();
  key_bias_g = key_bias_g.contiguous();
  adm_g = adm_g.contiguous();
  allow = allow.contiguous();
  induct_allow = induct_allow.contiguous();
  has_elig = has_elig.contiguous();
  induct_elig = induct_elig.contiguous();

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t N = kb.size(1);
  const int64_t C = cocktail.size(2);

  auto opts = q.options().dtype(torch::kFloat32);
  torch::Tensor out = torch::zeros({B, T, H}, opts);

  const float* qp = q.data_ptr<float>();
  const float* kbp = kb.data_ptr<float>();
  const float* wbp = wb.data_ptr<float>();
  const float* wnp = wb_next.data_ptr<float>();
  const float* cp = cocktail.data_ptr<float>();
  const float* cbp = cb.data_ptr<float>();
  const float* kbg = key_bias_g.data_ptr<float>();
  const float* adm = adm_g.data_ptr<float>();
  const bool* allowp = allow.data_ptr<bool>();
  const bool* iallowp = induct_allow.data_ptr<bool>();
  const bool* eligp = has_elig.data_ptr<bool>();
  const bool* ieligp = induct_elig.data_ptr<bool>();
  float* op = out.data_ptr<float>();

  const float scale = static_cast<float>(scale_d);
  const float cocktail_gain = static_cast<float>(cocktail_gain_d);
  const float short_gain = static_cast<float>(short_gain_d);
  const float pad_gain = static_cast<float>(pad_gain_d);
  const float induct_gain = static_cast<float>(induct_gain_d);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  #pragma omp parallel for collapse(2) schedule(static)
  for (int64_t bi = 0; bi < B; ++bi) {
    for (int64_t ti = 0; ti < T; ++ti) {
      const float* qbt = qp + (bi * T + ti) * H;
      const float* cbt = cp + (bi * T + ti) * C;
      float* obt = op + (bi * T + ti) * H;

      auto do_read = [&](bool perm, const float* values, const bool* mask, bool eligible, float gain) {
        if (!eligible || gain == 0.0f) return;
        float max_score = neg_inf;
        for (int64_t ni = 0; ni < N; ++ni) {
          if (!mask[(bi * T + ti) * N + ni]) continue;
          const float* kbv = kbp + (bi * N + ni) * H;
          float dot = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) dot += qbt[hi] * kbv[hi];
          float score = dot * scale + kbg[bi * N + ni] + adm[bi * N + ni];
          if (perm) {
            const float* cbv = cbp + (bi * N + ni) * C;
            float cdot = 0.0f;
            for (int64_t ci = 0; ci < C; ++ci) cdot += cbt[ci] * cbv[ci];
            score += cdot * cocktail_gain;
          }
          max_score = std::max(max_score, score);
        }
        if (max_score == neg_inf) return;
        float denom = 0.0f;
        for (int64_t ni = 0; ni < N; ++ni) {
          if (!mask[(bi * T + ti) * N + ni]) continue;
          const float* kbv = kbp + (bi * N + ni) * H;
          float dot = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) dot += qbt[hi] * kbv[hi];
          float score = dot * scale + kbg[bi * N + ni] + adm[bi * N + ni];
          if (perm) {
            const float* cbv = cbp + (bi * N + ni) * C;
            float cdot = 0.0f;
            for (int64_t ci = 0; ci < C; ++ci) cdot += cbt[ci] * cbv[ci];
            score += cdot * cocktail_gain;
          }
          denom += std::exp(score - max_score);
        }
        if (denom <= 0.0f) return;
        for (int64_t ni = 0; ni < N; ++ni) {
          if (!mask[(bi * T + ti) * N + ni]) continue;
          const float* kbv = kbp + (bi * N + ni) * H;
          float dot = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) dot += qbt[hi] * kbv[hi];
          float score = dot * scale + kbg[bi * N + ni] + adm[bi * N + ni];
          if (perm) {
            const float* cbv = cbp + (bi * N + ni) * C;
            float cdot = 0.0f;
            for (int64_t ci = 0; ci < C; ++ci) cdot += cbt[ci] * cbv[ci];
            score += cdot * cocktail_gain;
          }
          const float weight = gain * std::exp(score - max_score) / denom;
          const float* val = values + (bi * N + ni) * H;
          for (int64_t hi = 0; hi < H; ++hi) obt[hi] += weight * val[hi];
        }
      };

      do_read(false, wbp, allowp, eligp[bi * T + ti], short_gain);
      do_read(true, wbp, allowp, eligp[bi * T + ti], pad_gain);
      do_read(false, wnp, iallowp, ieligp[bi * T + ti], induct_gain);
    }
  }
  return out;
}

torch::Tensor linrec_forward_cpu(
    torch::Tensor a, torch::Tensor g, torch::Tensor y_init) {
  a = a.contiguous();
  g = g.contiguous();
  y_init = y_init.contiguous();
  TORCH_CHECK(a.scalar_type() == g.scalar_type() && a.scalar_type() == y_init.scalar_type(),
              "linrec_forward_cpu expects a/g/y_init to have the same dtype");
  const int64_t B = a.size(0), T = a.size(1), H = a.size(2);
  auto out = torch::empty_like(a);
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, a.scalar_type(), "linrec_forward_cpu", [&] {
    linrec_forward_impl<scalar_t>(
        a.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), y_init.data_ptr<scalar_t>(),
        out.data_ptr<scalar_t>(), B, T, H);
  });
  return out;
}

std::vector<torch::Tensor> linrec_backward_cpu(
    torch::Tensor a, torch::Tensor g, torch::Tensor y_init, torch::Tensor gy) {
  a = a.contiguous();
  g = g.contiguous();
  y_init = y_init.contiguous();
  gy = gy.contiguous();
  TORCH_CHECK(a.scalar_type() == g.scalar_type() && a.scalar_type() == y_init.scalar_type() &&
              a.scalar_type() == gy.scalar_type(),
              "linrec_backward_cpu expects a/g/y_init/gy to have the same dtype");
  const int64_t B = a.size(0), T = a.size(1), H = a.size(2);
  auto da = torch::zeros_like(a);
  auto dg = torch::zeros_like(g);
  auto dy = torch::zeros_like(y_init);
  AT_DISPATCH_FLOATING_TYPES_AND2(torch::kHalf, torch::kBFloat16, a.scalar_type(), "linrec_backward_cpu", [&] {
    linrec_backward_impl<scalar_t>(
        a.data_ptr<scalar_t>(), g.data_ptr<scalar_t>(), y_init.data_ptr<scalar_t>(), gy.data_ptr<scalar_t>(),
        da.data_ptr<scalar_t>(), dg.data_ptr<scalar_t>(), dy.data_ptr<scalar_t>(),
        B, T, H);
  });
  return {da, dg, dy};
}

std::vector<torch::Tensor> three_way_read_bwd(
    torch::Tensor grad_out, torch::Tensor q, torch::Tensor kb,
    torch::Tensor wb, torch::Tensor wb_next, torch::Tensor cocktail, torch::Tensor cb,
    torch::Tensor key_bias_g, torch::Tensor adm_g,
    torch::Tensor allow, torch::Tensor induct_allow,
    torch::Tensor has_elig, torch::Tensor induct_elig,
    double scale_d, double cocktail_gain_d,
    double short_gain_d, double pad_gain_d, double induct_gain_d) {

  grad_out = grad_out.contiguous();
  q = q.contiguous();
  kb = kb.contiguous();
  wb = wb.contiguous();
  wb_next = wb_next.contiguous();
  cocktail = cocktail.contiguous();
  cb = cb.contiguous();
  key_bias_g = key_bias_g.contiguous();
  adm_g = adm_g.contiguous();
  allow = allow.contiguous();
  induct_allow = induct_allow.contiguous();
  has_elig = has_elig.contiguous();
  induct_elig = induct_elig.contiguous();

  const int64_t B = q.size(0);
  const int64_t T = q.size(1);
  const int64_t H = q.size(2);
  const int64_t N = kb.size(1);
  const int64_t C = cocktail.size(2);

  auto opts = q.options().dtype(torch::kFloat32);
  torch::Tensor gq = torch::zeros({B, T, H}, opts);
  torch::Tensor gkb = torch::zeros({B, N, H}, opts);
  torch::Tensor gwb = torch::zeros({B, N, H}, opts);
  torch::Tensor gwb_next = torch::zeros({B, N, H}, opts);
  torch::Tensor gcocktail = torch::zeros({B, T, C}, opts);
  torch::Tensor gcb = torch::zeros({B, N, C}, opts);
  torch::Tensor gkbg = torch::zeros({B, N}, opts);
  torch::Tensor gadm = torch::zeros({B, N}, opts);
  torch::Tensor gscale_b = torch::zeros({B}, opts);
  torch::Tensor gcocktail_gain_b = torch::zeros({B}, opts);
  torch::Tensor gshort_b = torch::zeros({B}, opts);
  torch::Tensor gpad_b = torch::zeros({B}, opts);
  torch::Tensor ginduct_b = torch::zeros({B}, opts);

  const float* gop = grad_out.data_ptr<float>();
  const float* qp = q.data_ptr<float>();
  const float* kbp = kb.data_ptr<float>();
  const float* wbp = wb.data_ptr<float>();
  const float* wnp = wb_next.data_ptr<float>();
  const float* cp = cocktail.data_ptr<float>();
  const float* cbp = cb.data_ptr<float>();
  const float* kbg = key_bias_g.data_ptr<float>();
  const float* adm = adm_g.data_ptr<float>();
  const bool* allowp = allow.data_ptr<bool>();
  const bool* iallowp = induct_allow.data_ptr<bool>();
  const bool* eligp = has_elig.data_ptr<bool>();
  const bool* ieligp = induct_elig.data_ptr<bool>();

  float* gqp = gq.data_ptr<float>();
  float* gkbp = gkb.data_ptr<float>();
  float* gwbp = gwb.data_ptr<float>();
  float* gwnp = gwb_next.data_ptr<float>();
  float* gcp = gcocktail.data_ptr<float>();
  float* gcbp = gcb.data_ptr<float>();
  float* gkbgp = gkbg.data_ptr<float>();
  float* gadmp = gadm.data_ptr<float>();
  float* gscalep = gscale_b.data_ptr<float>();
  float* gcocktail_gainp = gcocktail_gain_b.data_ptr<float>();
  float* gshortp = gshort_b.data_ptr<float>();
  float* gpadp = gpad_b.data_ptr<float>();
  float* ginductp = ginduct_b.data_ptr<float>();

  const float scale = static_cast<float>(scale_d);
  const float cocktail_gain = static_cast<float>(cocktail_gain_d);
  const float short_gain = static_cast<float>(short_gain_d);
  const float pad_gain = static_cast<float>(pad_gain_d);
  const float induct_gain = static_cast<float>(induct_gain_d);
  const float neg_inf = -std::numeric_limits<float>::infinity();

  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> weights(N, 0.0f);
    std::vector<float> go_dot_value(N, 0.0f);
    for (int64_t ti = 0; ti < T; ++ti) {
      const float* qbt = qp + (bi * T + ti) * H;
      const float* cbt = cp + (bi * T + ti) * C;
      const float* go = gop + (bi * T + ti) * H;
      float* gqbt = gqp + (bi * T + ti) * H;
      float* gcbt = gcp + (bi * T + ti) * C;

      auto do_read_bwd = [&](bool perm, const float* values, float* gvalues,
                             const bool* mask, bool eligible, float gain, float* ggain) {
        if (!eligible || gain == 0.0f) return;
        float max_score = neg_inf;
        for (int64_t ni = 0; ni < N; ++ni) {
          weights[ni] = 0.0f;
          go_dot_value[ni] = 0.0f;
          if (!mask[(bi * T + ti) * N + ni]) continue;
          const float* kbv = kbp + (bi * N + ni) * H;
          float kdot = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) kdot += qbt[hi] * kbv[hi];
          float score = kdot * scale + kbg[bi * N + ni] + adm[bi * N + ni];
          if (perm) {
            const float* cbv = cbp + (bi * N + ni) * C;
            float cdot = 0.0f;
            for (int64_t ci = 0; ci < C; ++ci) cdot += cbt[ci] * cbv[ci];
            score += cdot * cocktail_gain;
          }
          max_score = std::max(max_score, score);
        }
        if (max_score == neg_inf) return;

        float denom = 0.0f;
        for (int64_t ni = 0; ni < N; ++ni) {
          if (!mask[(bi * T + ti) * N + ni]) continue;
          const float* kbv = kbp + (bi * N + ni) * H;
          float kdot = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) kdot += qbt[hi] * kbv[hi];
          float score = kdot * scale + kbg[bi * N + ni] + adm[bi * N + ni];
          if (perm) {
            const float* cbv = cbp + (bi * N + ni) * C;
            float cdot = 0.0f;
            for (int64_t ci = 0; ci < C; ++ci) cdot += cbt[ci] * cbv[ci];
            score += cdot * cocktail_gain;
          }
          float e = std::exp(score - max_score);
          weights[ni] = e;
          denom += e;
        }
        if (denom <= 0.0f) return;

        float dot_go_read = 0.0f;
        for (int64_t ni = 0; ni < N; ++ni) {
          if (!mask[(bi * T + ti) * N + ni]) continue;
          weights[ni] /= denom;
          const float* val = values + (bi * N + ni) * H;
          float d = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) {
            d += go[hi] * val[hi];
            gvalues[(bi * N + ni) * H + hi] += gain * weights[ni] * go[hi];
          }
          go_dot_value[ni] = d;
          dot_go_read += weights[ni] * d;
        }
        ggain[bi] += dot_go_read;

        for (int64_t ni = 0; ni < N; ++ni) {
          if (!mask[(bi * T + ti) * N + ni]) continue;
          const float dz = gain * weights[ni] * (go_dot_value[ni] - dot_go_read);
          const float* kbv = kbp + (bi * N + ni) * H;
          float kdot = 0.0f;
          for (int64_t hi = 0; hi < H; ++hi) {
            kdot += qbt[hi] * kbv[hi];
            gqbt[hi] += dz * scale * kbv[hi];
            gkbp[(bi * N + ni) * H + hi] += dz * scale * qbt[hi];
          }
          gkbgp[bi * N + ni] += dz;
          gadmp[bi * N + ni] += dz;
          gscalep[bi] += dz * kdot;
          if (perm) {
            const float* cbv = cbp + (bi * N + ni) * C;
            float cdot = 0.0f;
            for (int64_t ci = 0; ci < C; ++ci) {
              cdot += cbt[ci] * cbv[ci];
              gcbt[ci] += dz * cocktail_gain * cbv[ci];
              gcbp[(bi * N + ni) * C + ci] += dz * cocktail_gain * cbt[ci];
            }
            gcocktail_gainp[bi] += dz * cdot;
          }
        }
      };

      do_read_bwd(false, wbp, gwbp, allowp, eligp[bi * T + ti], short_gain, gshortp);
      do_read_bwd(true, wbp, gwbp, allowp, eligp[bi * T + ti], pad_gain, gpadp);
      do_read_bwd(false, wnp, gwnp, iallowp, ieligp[bi * T + ti], induct_gain, ginductp);
    }
  }

  return {
      gq, gkb, gwb, gwb_next, gcocktail, gcb, gkbg, gadm,
      gscale_b.sum().reshape({}), gcocktail_gain_b.sum().reshape({}),
      gshort_b.sum().reshape({}), gpad_b.sum().reshape({}), ginduct_b.sum().reshape({})
  };
}

// Native CPU training path. Forward saves the carried state entering each step;
// backward uses a reverse-time scan through the coupled recurrence. Its math
// mirrors dabsn_core_scan_cpu exactly.

// Forward-train: same recurrence as dabsn_core_scan_cpu, additionally returning the
// PRE-update carried states (bpre, epre, cpre) needed to recompute each step in
// backward. Returns {U, novel, p, ay, write, energy(=epre), saturation(=cpost),
//                     bpre, cpre}.
std::vector<torch::Tensor> dabsn_core_scan_fwd_train(
    torch::Tensor Wx, torch::Tensor Wgx, torch::Tensor GA_w,
    torch::Tensor beta, torch::Tensor log_kappa, torch::Tensor logit_recover,
    torch::Tensor k_s, torch::Tensor k_y, torch::Tensor k_b, torch::Tensor k_n, torch::Tensor k_bias,
    torch::Tensor r_s, torch::Tensor r_y, torch::Tensor r_b, torch::Tensor r_n, torch::Tensor r_bias,
    torch::Tensor logit_c_decay, torch::Tensor k_c, torch::Tensor r_c,
    torch::Tensor initial_b, torch::Tensor initial_e, torch::Tensor initial_c,
    double gate_alpha_d, double lam_d, double c_suppress_d) {

  Wx = Wx.contiguous(); Wgx = Wgx.contiguous(); GA_w = GA_w.contiguous();
  initial_b = initial_b.contiguous(); initial_e = initial_e.contiguous(); initial_c = initial_c.contiguous();
  const int64_t B = Wx.size(0), T = Wx.size(1), H = Wx.size(2);
  auto opts = Wx.options().dtype(torch::kFloat32);
  torch::Tensor U     = torch::empty({B, T, 2 * H}, opts);
  torch::Tensor novel = torch::empty({B, T, H}, opts);
  torch::Tensor pout  = torch::empty({B, T, H}, opts);
  torch::Tensor ayout = torch::empty({B, T, H}, opts);
  torch::Tensor wout  = torch::empty({B, T, H}, opts);
  torch::Tensor eout  = torch::empty({B, T, H}, opts);   // energy = pre-update e
  torch::Tensor cout  = torch::empty({B, T, H}, opts);   // saturation = post-update c
  torch::Tensor bpre  = torch::empty({B, T, H}, opts);   // b entering step t
  torch::Tensor cpre  = torch::empty({B, T, H}, opts);   // c entering step t
  torch::Tensor gatepre = torch::empty({B, T, H}, opts); // gate preactivation
  torch::Tensor bfinal = torch::empty({B, H}, opts);
  torch::Tensor efinal = torch::empty({B, H}, opts);
  torch::Tensor cfinal = torch::empty({B, H}, opts);

  const float* Wxp  = Wx.data_ptr<float>();
  const float* Wgxp = Wgx.data_ptr<float>();
  const float* GA   = GA_w.data_ptr<float>();
  const float* betap = beta.data_ptr<float>();
  const float* logk  = log_kappa.data_ptr<float>();
  const float* logr  = logit_recover.data_ptr<float>();
  const float* ks = k_s.data_ptr<float>(); const float* ky = k_y.data_ptr<float>();
  const float* kb = k_b.data_ptr<float>(); const float* kn = k_n.data_ptr<float>();
  const float* kbias = k_bias.data_ptr<float>();
  const float* rs = r_s.data_ptr<float>(); const float* ry = r_y.data_ptr<float>();
  const float* rb = r_b.data_ptr<float>(); const float* rn = r_n.data_ptr<float>();
  const float* rbias = r_bias.data_ptr<float>();
  const float* lcd = logit_c_decay.data_ptr<float>();
  const float* kcp = k_c.data_ptr<float>(); const float* rcp = r_c.data_ptr<float>();
  const float* Binit = initial_b.data_ptr<float>();
  const float* Einit = initial_e.data_ptr<float>();
  const float* Cinit = initial_c.data_ptr<float>();

  float* Up = U.data_ptr<float>(); float* Np = novel.data_ptr<float>();
  float* Pp = pout.data_ptr<float>(); float* Ap = ayout.data_ptr<float>();
  float* Wp = wout.data_ptr<float>(); float* Ep = eout.data_ptr<float>();
  float* Cp = cout.data_ptr<float>(); float* Bpre = bpre.data_ptr<float>();
  float* Cpre = cpre.data_ptr<float>(); float* Gatepre = gatepre.data_ptr<float>();
  float* Bfinal = bfinal.data_ptr<float>();
  float* Efinal = efinal.data_ptr<float>();
  float* Cfinal = cfinal.data_ptr<float>();

  const float gate_alpha = (float)gate_alpha_d, lam = (float)lam_d, c_suppress = (float)c_suppress_d;

  if (use_local_field_blas_shape(B, T)) {
    torch::Tensor b = initial_b.clone();
    torch::Tensor e = initial_e.clone();
    torch::Tensor c = initial_c.clone();
    torch::Tensor GA_t = GA_w.transpose(0, 1).contiguous();
    torch::Tensor gate_alpha_t = torch::tensor(gate_alpha, opts);
    torch::Tensor lam_t = torch::tensor(lam, opts);
    torch::Tensor c_suppress_t = torch::tensor(c_suppress, opts);
    torch::Tensor kappa = torch::nn::functional::softplus(log_kappa);
    torch::Tensor recover = torch::sigmoid(logit_recover);
    torch::Tensor c_decay = torch::sigmoid(logit_c_decay);

    for (int64_t t = 0; t < T; ++t) {
      torch::Tensor wx = Wx.select(1, t);
      torch::Tensor wgx = Wgx.select(1, t);
      bpre.select(1, t).copy_(b);
      cpre.select(1, t).copy_(c);
      eout.select(1, t).copy_(e);
      torch::Tensor y = torch::tanh(wx + b);
      torch::Tensor ugay = torch::matmul(y, GA_t);
      torch::Tensor ug = ugay.slice(1, 0, H);
      torch::Tensor ay = ugay.slice(1, H, 2 * H);
      torch::Tensor gate_pre = wgx + ug;
      gatepre.select(1, t).copy_(gate_pre);
      torch::Tensor s = torch::clamp(gate_pre + 3.0, 0.0, 6.0) * (1.0 / 6.0);
      torch::Tensor novelty = torch::tanh(torch::abs(ay - b));
      torch::Tensor stress = novelty * (1.0 - e);
      c = c_decay * c + (1.0 - c_decay) * stress;
      torch::Tensor novelty_eff = novelty * (1.0 - c_suppress_t * c);
      torch::Tensor energy_nov = novelty_eff;
      torch::Tensor tb = torch::tanh(b);
      torch::Tensor k_sig = k_s * s + k_y * y + k_b * tb + k_n * energy_nov + k_bias;
      torch::Tensor r_sig = r_s * s + r_y * y + r_b * tb + r_n * energy_nov + r_bias;
      k_sig = k_sig + k_c * c;
      r_sig = r_sig + r_c * c;
      torch::Tensor k_t = kappa * torch::exp(0.5 * torch::tanh(k_sig));
      torch::Tensor r_t = torch::clamp(recover * torch::exp(0.5 * torch::tanh(r_sig)), 0.0, 1.0);
      torch::Tensor p = s * e;
      torch::Tensor write = p * ay;
      torch::Tensor bn = (1.0 - gate_alpha_t) * b + beta + lam_t * write;
      torch::Tensor en = torch::clamp(e + r_t * (1.0 - e) - k_t * p, 0.0, 1.0);

      torch::Tensor Ut = U.select(1, t);
      Ut.slice(1, 0, H).copy_(y);
      Ut.slice(1, H, 2 * H).copy_(bn);
      novel.select(1, t).copy_(novelty);
      pout.select(1, t).copy_(p);
      ayout.select(1, t).copy_(ay);
      wout.select(1, t).copy_(write);
      cout.select(1, t).copy_(c);
      b = bn;
      e = en;
    }
    return {U, novel, pout, ayout, wout, eout, cout, bpre, cpre, b, e, c, gatepre};
  }

  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> b(H), e(H), c(H);
    for (int64_t i = 0; i < H; ++i) {
      b[i] = Binit[bi * H + i];
      e[i] = Einit[bi * H + i];
      c[i] = Cinit[bi * H + i];
    }
    std::vector<float> y(H), ug(H), ay(H);
    for (int64_t t = 0; t < T; ++t) {
      const float* wx  = Wxp  + (bi * T + t) * H;
      const float* wgx = Wgxp + (bi * T + t) * H;
      for (int64_t i = 0; i < H; ++i) y[i] = std::tanh(wx[i] + b[i]);
      for (int64_t i = 0; i < H; ++i) {
        const float* gu = GA + i * H;
        const float* ga = GA + (H + i) * H;
        float su = 0.0f, sa = 0.0f;
        for (int64_t j = 0; j < H; ++j) { su += gu[j] * y[j]; sa += ga[j] * y[j]; }
        ug[i] = su; ay[i] = sa;
      }
      float* up = Up + (bi * T + t) * 2 * H;
      int64_t off = (bi * T + t) * H;
      for (int64_t i = 0; i < H; ++i) {
        float b0 = b[i], e0 = e[i], c0 = c[i];
        Bpre[off + i] = b0; Cpre[off + i] = c0; Ep[off + i] = e0;
        float gate_pre = wgx[i] + ug[i];
        Gatepre[off + i] = gate_pre;
        float s = hardsigmoidf(gate_pre);
        float novelty = std::tanh(std::fabs(ay[i] - b0));
        float c_decay = sigmoidf(lcd[i]);
        float stress = novelty * (1.0f - e0);
        float cpost = c_decay * c0 + (1.0f - c_decay) * stress;
        float novelty_eff = novelty * (1.0f - c_suppress * cpost);
        float energy_nov = novelty_eff;
        float tb = std::tanh(b0);
        float k_sig = ks[i] * s + ky[i] * y[i] + kb[i] * tb + kn[i] * energy_nov + kbias[i];
        float r_sig = rs[i] * s + ry[i] * y[i] + rb[i] * tb + rn[i] * energy_nov + rbias[i];
        k_sig += kcp[i] * cpost; r_sig += rcp[i] * cpost;
        float kappa = softplusf(logk[i]), recover = sigmoidf(logr[i]);
        float k_t = kappa * std::exp(0.5f * std::tanh(k_sig));
        float r_t = std::min(std::max(recover * std::exp(0.5f * std::tanh(r_sig)), 0.0f), 1.0f);
        float p = s * e0;
        float write = p * ay[i];
        float bn = (1.0f - gate_alpha) * b0 + betap[i] + lam * (p * ay[i]);
        float en = std::min(std::max(e0 + r_t * (1.0f - e0) - k_t * p, 0.0f), 1.0f);
        up[i] = y[i]; up[H + i] = bn;
        Np[off + i] = novelty; Pp[off + i] = p; Ap[off + i] = ay[i]; Wp[off + i] = write;
        Cp[off + i] = cpost;
        b[i] = bn; e[i] = en; c[i] = cpost;
      }
    }
    for (int64_t i = 0; i < H; ++i) {
      Bfinal[bi * H + i] = b[i];
      Efinal[bi * H + i] = e[i];
      Cfinal[bi * H + i] = c[i];
    }
  }
  return {U, novel, pout, ayout, wout, eout, cout, bpre, cpre, bfinal, efinal, cfinal, gatepre};
}

// Backward: reverse-time scan. Grad inputs are for the 7 forward outputs
// {U, novel, p, ay, write, energy, saturation}; saved tensors are {bpre, epre, cpre}
// plus all forward inputs. Returns grads for {Wx, Wgx, GA_w, beta, log_kappa,
// logit_recover, k_s,k_y,k_b,k_n,k_bias, r_s,r_y,r_b,r_n,r_bias, logit_c_decay,
// k_c, r_c, gate_alpha, lam, c_suppress} (the 3 scalars as 0-dim tensors).
std::vector<torch::Tensor> dabsn_core_scan_bwd(
    torch::Tensor gU, torch::Tensor gNov, torch::Tensor gP, torch::Tensor gAy,
    torch::Tensor gWrite, torch::Tensor gEnergy, torch::Tensor gCort,
    torch::Tensor gFinalB, torch::Tensor gFinalE, torch::Tensor gFinalC,
    torch::Tensor bpre, torch::Tensor epre, torch::Tensor cpre,
    torch::Tensor Wx, torch::Tensor Wgx, torch::Tensor GA_w,
    torch::Tensor beta, torch::Tensor log_kappa, torch::Tensor logit_recover,
    torch::Tensor k_s, torch::Tensor k_y, torch::Tensor k_b, torch::Tensor k_n, torch::Tensor k_bias,
    torch::Tensor r_s, torch::Tensor r_y, torch::Tensor r_b, torch::Tensor r_n, torch::Tensor r_bias,
    torch::Tensor logit_c_decay, torch::Tensor k_c, torch::Tensor r_c,
    double gate_alpha_d, double lam_d, double c_suppress_d) {

  gU = gU.contiguous(); gNov = gNov.contiguous(); gP = gP.contiguous();
  gAy = gAy.contiguous(); gWrite = gWrite.contiguous(); gEnergy = gEnergy.contiguous();
  gCort = gCort.contiguous();
  gFinalB = gFinalB.contiguous(); gFinalE = gFinalE.contiguous(); gFinalC = gFinalC.contiguous();
  Wx = Wx.contiguous(); Wgx = Wgx.contiguous(); GA_w = GA_w.contiguous();
  bpre = bpre.contiguous(); epre = epre.contiguous(); cpre = cpre.contiguous();

  const int64_t B = Wx.size(0), T = Wx.size(1), H = Wx.size(2);
  auto opts = Wx.options().dtype(torch::kFloat32);

  torch::Tensor dWx  = torch::zeros({B, T, H}, opts);
  torch::Tensor dWgx = torch::zeros({B, T, H}, opts);
  torch::Tensor dGA_b = torch::zeros({B, 2 * H, H}, opts);   // per-batch, reduced at end
  torch::Tensor dpar_b = torch::zeros({B, 17, H}, opts);     // per-batch param-vector grads
  torch::Tensor dscal_b = torch::zeros({B, 3}, opts);        // alpha, lam, c_suppress
  torch::Tensor dInitialB = torch::empty({B, H}, opts);
  torch::Tensor dInitialE = torch::empty({B, H}, opts);
  torch::Tensor dInitialC = torch::empty({B, H}, opts);

  const float gate_alpha = (float)gate_alpha_d, lam = (float)lam_d, c_suppress = (float)c_suppress_d;

  const float* Wxp = Wx.data_ptr<float>(); const float* Wgxp = Wgx.data_ptr<float>();
  const float* GA = GA_w.data_ptr<float>();
  const float* logk = log_kappa.data_ptr<float>(); const float* logr = logit_recover.data_ptr<float>();
  const float* ks = k_s.data_ptr<float>(); const float* ky = k_y.data_ptr<float>();
  const float* kb = k_b.data_ptr<float>(); const float* kn = k_n.data_ptr<float>();
  const float* kbias = k_bias.data_ptr<float>();
  const float* rs = r_s.data_ptr<float>(); const float* ry = r_y.data_ptr<float>();
  const float* rb = r_b.data_ptr<float>(); const float* rn = r_n.data_ptr<float>();
  const float* rbias = r_bias.data_ptr<float>();
  const float* lcd = logit_c_decay.data_ptr<float>();
  const float* kcp = k_c.data_ptr<float>(); const float* rcp = r_c.data_ptr<float>();
  const float* Bpre = bpre.data_ptr<float>(); const float* Epre = epre.data_ptr<float>();
  const float* Cpre = cpre.data_ptr<float>();
  const float* gUp = gU.data_ptr<float>(); const float* gNp = gNov.data_ptr<float>();
  const float* gPp = gP.data_ptr<float>(); const float* gAp = gAy.data_ptr<float>();
  const float* gWp = gWrite.data_ptr<float>(); const float* gEp = gEnergy.data_ptr<float>();
  const float* gCp = gCort.data_ptr<float>();
  const float* gFBp = gFinalB.data_ptr<float>();
  const float* gFEp = gFinalE.data_ptr<float>();
  const float* gFCp = gFinalC.data_ptr<float>();

  float* dWxp = dWx.data_ptr<float>(); float* dWgxp = dWgx.data_ptr<float>();
  float* dGAp = dGA_b.data_ptr<float>(); float* dparp = dpar_b.data_ptr<float>();
  float* dscp = dscal_b.data_ptr<float>();
  float* dIBp = dInitialB.data_ptr<float>();
  float* dIEp = dInitialE.data_ptr<float>();
  float* dICp = dInitialC.data_ptr<float>();
  // dpar rows: 0 beta,1 log_kappa,2 logit_recover,3 k_s,4 k_y,5 k_b,6 k_n,7 k_bias,
  //            8 r_s,9 r_y,10 r_b,11 r_n,12 r_bias,13 logit_c_decay,14 k_c,15 r_c, (16 unused)

  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> gb(H), ge(H), gc(H);   // carried adjoints (from t+1)
    for (int64_t i = 0; i < H; ++i) {
      gb[i] = gFBp[bi * H + i];
      ge[i] = gFEp[bi * H + i];
      gc[i] = gFCp[bi * H + i];
    }
    std::vector<float> y(H), ug(H), ay(H), gugv(H), gayv(H), gyv(H);
    float* dGA = dGAp + bi * 2 * H * H;
    float* dpar = dparp + bi * 17 * H;
    for (int64_t t = T - 1; t >= 0; --t) {
      const float* wx  = Wxp  + (bi * T + t) * H;
      const float* wgx = Wgxp + (bi * T + t) * H;
      int64_t off = (bi * T + t) * H;
      // recompute y, ug, ay for this step
      for (int64_t i = 0; i < H; ++i) y[i] = std::tanh(wx[i] + Bpre[off + i]);
      for (int64_t i = 0; i < H; ++i) {
        const float* gu = GA + i * H; const float* ga = GA + (H + i) * H;
        float su = 0.0f, sa = 0.0f;
        for (int64_t j = 0; j < H; ++j) { su += gu[j] * y[j]; sa += ga[j] * y[j]; }
        ug[i] = su; ay[i] = sa;
      }
      for (int64_t i = 0; i < H; ++i) {
        float b0 = Bpre[off + i], e0 = Epre[off + i], c0 = Cpre[off + i];
        // ---- recompute forward locals ----
        float xs = wgx[i] + ug[i];
        float s = hardsigmoidf(xs);
        float hsp = (xs > -3.0f && xs < 3.0f) ? (1.0f / 6.0f) : 0.0f;
        float dnv = ay[i] - b0; float adn = std::fabs(dnv);
        float sgn = (dnv > 0.0f) ? 1.0f : ((dnv < 0.0f) ? -1.0f : 0.0f);
        float novelty = std::tanh(adn);
        float c_decay = sigmoidf(lcd[i]);
        float stress = novelty * (1.0f - e0);
        float cpost = c_decay * c0 + (1.0f - c_decay) * stress;
        float novelty_eff = novelty * (1.0f - c_suppress * cpost);
        float energy_nov = novelty_eff;
        float tb = std::tanh(b0);
        float k_sig = ks[i]*s + ky[i]*y[i] + kb[i]*tb + kn[i]*energy_nov + kbias[i];
        float r_sig = rs[i]*s + ry[i]*y[i] + rb[i]*tb + rn[i]*energy_nov + rbias[i];
        k_sig += kcp[i]*cpost; r_sig += rcp[i]*cpost;
        float kappa = softplusf(logk[i]); float kappa_p = sigmoidf(logk[i]);
        float recover = sigmoidf(logr[i]);
        float tk = std::tanh(k_sig); float ek = std::exp(0.5f * tk); float k_t = kappa * ek;
        float tr = std::tanh(r_sig); float er = std::exp(0.5f * tr); float r_pre = recover * er;
        float r_t = std::min(std::max(r_pre, 0.0f), 1.0f);
        float rclamp = (r_pre > 0.0f && r_pre < 1.0f) ? 1.0f : 0.0f;
        float p = s * e0;
        float en_pre = e0 + r_t * (1.0f - e0) - k_t * p;
        float eclamp = (en_pre > 0.0f && en_pre < 1.0f) ? 1.0f : 0.0f;

        // ---- incoming adjoints ----
        float gbn = gUp[(bi*T+t)*2*H + H + i] + gb[i];   // U b-part + carried b
        float gen = ge[i];                                // en -> next e_prev only
        float gcpost = gCp[off + i] + gc[i];              // saturation output + carried c
        float ge0 = gEp[off + i];                         // energy output = e0
        float gy = gUp[(bi*T+t)*2*H + i];                 // U y-part
        float gay = gAp[off + i];                         // ay output
        float gb0 = 0.0f, gc0 = 0.0f;
        float gtb = 0.0f, gs = 0.0f, gnov_acc = gNp[off + i];
        float genergy_nov = 0.0f;

        // en = clamp(en_pre): en_pre = e0 + r_t*(1-e0) - k_t*p
        float den_pre = gen * eclamp;
        ge0 += den_pre * (1.0f - r_t);
        float gr_t = den_pre * (1.0f - e0);
        float gk_t = -den_pre * p;
        float gp = -den_pre * k_t;
        // bn = (1-alpha)*b0 + beta + lam*(p*ay)
        gb0 += gbn * (1.0f - gate_alpha);
        dpar[0*H + i] += gbn;                              // d beta
        dscp[bi*3 + 1] += gbn * (p * ay[i]);              // d lam
        dscp[bi*3 + 0] += gbn * (-b0);                    // d gate_alpha
        float gwrite = gWp[off + i] + gbn * lam;          // write output + bn path
        // write = p*ay
        gp += gwrite * ay[i];
        gay += gwrite * p;
        // p = s*e0
        gp += gPp[off + i];
        gs += gp * e0;
        ge0 += gp * s;
        // k_t = kappa*ek
        dpar[1*H + i] += (gk_t * ek) * kappa_p;            // d log_kappa
        float gk_sig = gk_t * kappa * ek * 0.5f * (1.0f - tk*tk);
        // r_t = clamp(recover*er)
        float gr_pre = gr_t * rclamp;
        dpar[2*H + i] += (gr_pre * er) * (recover * (1.0f - recover));  // d logit_recover
        float gr_sig = gr_pre * recover * er * 0.5f * (1.0f - tr*tr);
        // k_sig terms
        dpar[3*H + i] += gk_sig * s;   gs += gk_sig * ks[i];
        dpar[4*H + i] += gk_sig * y[i]; gy += gk_sig * ky[i];
        dpar[5*H + i] += gk_sig * tb;  gtb += gk_sig * kb[i];
        dpar[6*H + i] += gk_sig * energy_nov; genergy_nov += gk_sig * kn[i];
        dpar[7*H + i] += gk_sig;       // k_bias
        // r_sig terms
        dpar[8*H + i] += gr_sig * s;   gs += gr_sig * rs[i];
        dpar[9*H + i] += gr_sig * y[i]; gy += gr_sig * ry[i];
        dpar[10*H + i] += gr_sig * tb; gtb += gr_sig * rb[i];
        dpar[11*H + i] += gr_sig * energy_nov; genergy_nov += gr_sig * rn[i];
        dpar[12*H + i] += gr_sig;      // r_bias
        dpar[14*H + i] += gk_sig * cpost; gcpost += gk_sig * kcp[i];
        dpar[15*H + i] += gr_sig * cpost; gcpost += gr_sig * rcp[i];
        float gnovelty_eff = genergy_nov;
        // tb = tanh(b0)
        gb0 += gtb * (1.0f - tb*tb);
        // novelty_eff / saturation
        gnov_acc += gnovelty_eff * (1.0f - c_suppress * cpost);
        gcpost += gnovelty_eff * novelty * (-c_suppress);
        dscp[bi*3 + 2] += gnovelty_eff * novelty * (-cpost);   // d c_suppress
        // cpost = c_decay*c0 + (1-c_decay)*stress
        gc0 += gcpost * c_decay;
        float gstress = gcpost * (1.0f - c_decay);
        float gcdec = gcpost * (c0 - stress);
        dpar[13*H + i] += gcdec * c_decay * (1.0f - c_decay);  // d logit_c_decay
        gnov_acc += gstress * (1.0f - e0);
        ge0 += gstress * (-novelty);
        // novelty = tanh(|ay-b0|)
        float gd = gnov_acc * (1.0f - novelty*novelty) * sgn;
        gay += gd;
        gb0 += -gd;
        // s = hardsigmoid(wgx+ug)
        float gx_s = gs * hsp;
        dWgxp[off + i] += gx_s;
        gugv[i] = gx_s;       // ug only feeds s
        gayv[i] = gay;        // total grad on ay (for matvec)
        gyv[i] = gy;          // partial grad on y (pre-matvec); matvec adds below
        // stash carry pieces via temporaries: store gb0/gc0/ge0 into carry after matvec
        gb[i] = gb0;          // reuse gb as gb0 accumulator (matvec adds y-path next)
        ge[i] = ge0;          // e carry for previous step (en path already none; this is e0)
        gc[i] = gc0;          // c carry for previous step
      }
      // matvec backward: gy += Ug^T @ gug + A^T @ gay ; dGA += outer(gug,y)/outer(gay,y)
      for (int64_t i = 0; i < H; ++i) {
        float gu_i = gugv[i], ga_i = gayv[i];
        float* dGu = dGA + i * H;
        float* dGa = dGA + (H + i) * H;
        for (int64_t j = 0; j < H; ++j) {
          dGu[j] += gu_i * y[j];
          dGa[j] += ga_i * y[j];
          gyv[j] += GA[i*H + j] * gu_i + GA[(H+i)*H + j] * ga_i;
        }
      }
      // y = tanh(wx + b0): finish dWx and add y-path to b0 carry
      for (int64_t i = 0; i < H; ++i) {
        float gpre_y = gyv[i] * (1.0f - y[i]*y[i]);
        dWxp[off + i] += gpre_y;
        gb[i] += gpre_y;       // b0 total carried to step t-1
      }
    }
    for (int64_t i = 0; i < H; ++i) {
      dIBp[bi * H + i] = gb[i];
      dIEp[bi * H + i] = ge[i];
      dICp[bi * H + i] = gc[i];
    }
  }

  torch::Tensor dGA_w = dGA_b.sum(0);
  torch::Tensor dpar = dpar_b.sum(0);     // (17, H)
  torch::Tensor dscal = dscal_b.sum(0);   // (3,)
  return {dWx, dWgx, dGA_w,
          dpar[0], dpar[1], dpar[2],
          dpar[3], dpar[4], dpar[5], dpar[6], dpar[7],
          dpar[8], dpar[9], dpar[10], dpar[11], dpar[12],
          dpar[13], dpar[14], dpar[15],
          dInitialB, dInitialE, dInitialC,
          dscal[0], dscal[1], dscal[2]};
}

// Forward-time online temporal-credit replay. This consumes only saved local
// values and current-step output cotangents. It never performs reverse-time
// recurrence and never reconstructs the four forward matrix-vector products.
std::vector<torch::Tensor> dabsn_core_online_replay_cpu(
    torch::Tensor inputs, torch::Tensor bpre, torch::Tensor epre, torch::Tensor cpre,
    torch::Tensor y_tape, torch::Tensor ay_tape, torch::Tensor gate_pre_tape,
    torch::Tensor gy_out, torch::Tensor gbn_out, torch::Tensor gnov_out,
    torch::Tensor gp_out, torch::Tensor gay_out, torch::Tensor gwrite_out,
    torch::Tensor ge_out, torch::Tensor gcpost_out,
    torch::Tensor W, torch::Tensor A, torch::Tensor Wg, torch::Tensor Ug,
    torch::Tensor Wg_bias, torch::Tensor beta, torch::Tensor log_kappa,
    torch::Tensor logit_recover, torch::Tensor k_s, torch::Tensor k_y,
    torch::Tensor k_b, torch::Tensor k_n, torch::Tensor k_bias,
    torch::Tensor r_s, torch::Tensor r_y, torch::Tensor r_b, torch::Tensor r_n,
    torch::Tensor r_bias, torch::Tensor k_c, torch::Tensor r_c,
    torch::Tensor logit_alpha, torch::Tensor log_lambda,
    torch::Tensor logit_c_decay, torch::Tensor logit_c_suppress,
    torch::Tensor w_elig, torch::Tensor a_elig, torch::Tensor wg_elig,
    torch::Tensor ug_elig, torch::Tensor local_elig, torch::Tensor global_elig) {

  inputs = inputs.contiguous(); bpre = bpre.contiguous(); epre = epre.contiguous();
  cpre = cpre.contiguous(); y_tape = y_tape.contiguous(); ay_tape = ay_tape.contiguous();
  gate_pre_tape = gate_pre_tape.contiguous(); gy_out = gy_out.contiguous();
  gbn_out = gbn_out.contiguous(); gnov_out = gnov_out.contiguous();
  gp_out = gp_out.contiguous(); gay_out = gay_out.contiguous();
  gwrite_out = gwrite_out.contiguous(); ge_out = ge_out.contiguous();
  gcpost_out = gcpost_out.contiguous(); W = W.contiguous(); A = A.contiguous();
  Wg = Wg.contiguous(); Ug = Ug.contiguous(); Wg_bias = Wg_bias.contiguous();
  w_elig = w_elig.contiguous(); a_elig = a_elig.contiguous();
  wg_elig = wg_elig.contiguous(); ug_elig = ug_elig.contiguous();
  local_elig = local_elig.contiguous(); global_elig = global_elig.contiguous();

  TORCH_CHECK(inputs.scalar_type() == torch::kFloat32, "online CPU replay requires float32");
  const int64_t B = inputs.size(0), T = inputs.size(1), I = inputs.size(2);
  const int64_t H = y_tape.size(2);
  TORCH_CHECK(bpre.sizes() == torch::IntArrayRef({B, T, H}), "invalid budget tape shape");
  TORCH_CHECK(epre.sizes() == bpre.sizes() && cpre.sizes() == bpre.sizes(), "invalid state tape shape");
  TORCH_CHECK(W.sizes() == torch::IntArrayRef({H, I}), "invalid W shape");
  TORCH_CHECK(A.sizes() == torch::IntArrayRef({H, H}), "invalid A shape");
  TORCH_CHECK(Wg.sizes() == torch::IntArrayRef({H, I}), "invalid Wg shape");
  TORCH_CHECK(Ug.sizes() == torch::IntArrayRef({H, H}), "invalid Ug shape");
  TORCH_CHECK(w_elig.sizes() == torch::IntArrayRef({B, H, I}), "invalid W eligibility shape");
  TORCH_CHECK(a_elig.sizes() == torch::IntArrayRef({B, H, H}), "invalid A eligibility shape");
  TORCH_CHECK(wg_elig.sizes() == torch::IntArrayRef({B, H, I}), "invalid Wg eligibility shape");
  TORCH_CHECK(ug_elig.sizes() == torch::IntArrayRef({B, H, H}), "invalid Ug eligibility shape");
  TORCH_CHECK(local_elig.sizes() == torch::IntArrayRef({B, H, 3, 16}), "invalid local eligibility shape");
  TORCH_CHECK(global_elig.sizes() == torch::IntArrayRef({B, H, 3, 4}), "invalid global eligibility shape");

  auto opts = inputs.options().dtype(torch::kFloat32);
  torch::Tensor next_w_elig, next_a_elig, next_wg_elig, next_ug_elig;
  torch::Tensor next_local_elig = local_elig.clone();
  torch::Tensor next_global_elig = global_elig.clone();
  torch::Tensor grad_input = torch::zeros({B, T, I}, opts);
  torch::Tensor dW = torch::zeros({H, I}, opts);
  torch::Tensor dA = torch::zeros({H, H}, opts);
  torch::Tensor dWg = torch::zeros({H, I}, opts);
  torch::Tensor dUg = torch::zeros({H, H}, opts);
  torch::Tensor dWg_bias = torch::zeros({H}, opts);
  torch::Tensor dlocal = torch::zeros({16, H}, opts);
  torch::Tensor dglobal = torch::zeros({4}, opts);

  const float* xp = inputs.data_ptr<float>();
  const float* bp = bpre.data_ptr<float>(); const float* ep = epre.data_ptr<float>();
  const float* cp = cpre.data_ptr<float>(); const float* yp = y_tape.data_ptr<float>();
  const float* ayp = ay_tape.data_ptr<float>(); const float* gateprep = gate_pre_tape.data_ptr<float>();
  const float* gyop = gy_out.data_ptr<float>(); const float* gbnop = gbn_out.data_ptr<float>();
  const float* gnop = gnov_out.data_ptr<float>(); const float* gpop = gp_out.data_ptr<float>();
  const float* gayop = gay_out.data_ptr<float>(); const float* gwop = gwrite_out.data_ptr<float>();
  const float* geop = ge_out.data_ptr<float>(); const float* gcop = gcpost_out.data_ptr<float>();
  const float* Wp = W.data_ptr<float>(); const float* Ap = A.data_ptr<float>();
  const float* Wgp = Wg.data_ptr<float>(); const float* Ugp = Ug.data_ptr<float>();
  const float* Wgbp = Wg_bias.data_ptr<float>(); const float* betap = beta.data_ptr<float>();
  const float* logkp = log_kappa.data_ptr<float>(); const float* logrp = logit_recover.data_ptr<float>();
  const float* ksp = k_s.data_ptr<float>(); const float* kyp = k_y.data_ptr<float>();
  const float* kbp = k_b.data_ptr<float>(); const float* knp = k_n.data_ptr<float>();
  const float* kbiasp = k_bias.data_ptr<float>(); const float* rsp = r_s.data_ptr<float>();
  const float* ryp = r_y.data_ptr<float>(); const float* rbp = r_b.data_ptr<float>();
  const float* rnp = r_n.data_ptr<float>(); const float* rbiasp = r_bias.data_ptr<float>();
  const float* kcp = k_c.data_ptr<float>(); const float* rcp = r_c.data_ptr<float>();
  const float alpha_logit = logit_alpha.reshape({}).item<float>();
  const float lambda_log = log_lambda.reshape({}).item<float>();
  const float decay_logit = logit_c_decay.reshape({}).item<float>();
  const float suppress_logit = logit_c_suppress.reshape({}).item<float>();
  const float alpha = sigmoidf(alpha_logit), lam = softplusf(lambda_log);
  const float decay = sigmoidf(decay_logit), suppress = sigmoidf(suppress_logit);

  float* lep = next_local_elig.data_ptr<float>(); float* gep = next_global_elig.data_ptr<float>();
  float* dlp = dlocal.data_ptr<float>(); float* dgp = dglobal.data_ptr<float>();

  // Current-step VJPs are independent across B*T. Compute their matrix
  // products once with BLAS, rather than reconstructing every output-row
  // cotangent inside every input-channel iteration.
  auto gate_tensor = torch::clamp(gate_pre_tape / 6.0 + 0.5, 0.0, 1.0);
  auto novelty_tensor = torch::tanh(torch::abs(ay_tape - bpre));
  auto p_tensor = gate_tensor * epre;
  auto write_gradient = gwrite_out + gbn_out * lam;
  auto plastic_gradient = gp_out + write_gradient * ay_tape;
  auto novelty_gradient = gnov_out + gcpost_out * (1.0f - decay) * (1.0f - epre);
  auto difference_gradient = novelty_gradient * (1.0f - novelty_tensor.square()) * torch::sign(ay_tape - bpre);
  auto expression_gradient = gay_out + write_gradient * p_tensor + difference_gradient;
  auto gate_gradient = plastic_gradient * epre *
      ((gate_pre_tape > -3.0) & (gate_pre_tape < 3.0)).to(torch::kFloat32) * (1.0f / 6.0f);
  auto flat_expression = expression_gradient.reshape({B * T, H});
  auto flat_gate = gate_gradient.reshape({B * T, H});
  auto y_gradient = gy_out + (torch::mm(flat_expression, A) + torch::mm(flat_gate, Ug)).reshape({B, T, H});
  auto ypre_gradient = y_gradient * (1.0f - y_tape.square());
  auto incoming_budget = (gbn_out * (1.0f - alpha) - difference_gradient + ypre_gradient).contiguous();
  auto incoming_energy = (ge_out + plastic_gradient * gate_tensor - gcpost_out * (1.0f - decay) * novelty_tensor).contiguous();
  auto incoming_saturation = (gcpost_out * decay).contiguous();
  auto flat_x = inputs.reshape({B * T, I});
  auto flat_y = y_tape.reshape({B * T, H});
  auto flat_ypre = ypre_gradient.reshape({B * T, H});
  dW = torch::mm(flat_ypre.t(), flat_x);
  dA = torch::mm(flat_expression.t(), flat_y);
  dWg = torch::mm(flat_gate.t(), flat_x);
  dUg = torch::mm(flat_gate.t(), flat_y);
  dWg_bias = flat_gate.sum(0);
  grad_input = (torch::mm(flat_ypre, W) + torch::mm(flat_gate, Wg)).reshape({B, T, I});
  dlocal[1].copy_(gbn_out.sum(torch::IntArrayRef({0, 1})));
  dglobal[0].copy_((gbn_out * (-bpre) * alpha * (1.0f - alpha)).sum());
  dglobal[1].copy_((gbn_out * (p_tensor * ay_tape) * sigmoidf(lambda_log)).sum());
  dglobal[2].copy_((gcpost_out * (cpre - novelty_tensor * (1.0f - epre)) * decay * (1.0f - decay)).sum());
  const float* ibp = incoming_budget.data_ptr<float>();
  const float* iep = incoming_energy.data_ptr<float>();
  const float* icp = incoming_saturation.data_ptr<float>();
  auto transitions = torch::empty({B, T, H}, opts);
  float* jp = transitions.data_ptr<float>();

  for (int64_t bi = 0; bi < B; ++bi) {
    for (int64_t ti = 0; ti < T; ++ti) {
      const int64_t step_off = (bi * T + ti) * H;
      const int64_t x_off = (bi * T + ti) * I;
      for (int64_t h = 0; h < H; ++h) {
        const int64_t off = step_off + h;
        const float b0 = bp[off], e0 = ep[off], c0 = cp[off];
        const float y = yp[off], ay = ayp[off], gate_pre = gateprep[off];
        const float gate = hardsigmoidf(gate_pre);
        const float novelty = std::tanh(std::fabs(ay - b0));
        const float stress = novelty * (1.0f - e0);
        const float cpost = decay * c0 + (1.0f - decay) * stress;
        const float p = gate * e0;

        const float incoming[3] = {ibp[off], iep[off], icp[off]};
        const int64_t lbase = ((bi * H + h) * 3) * 16;
        const int64_t gbase = ((bi * H + h) * 3) * 4;
        for (int pindex = 0; pindex < 16; ++pindex) {
          float value = 0.0f;
          for (int state_index = 0; state_index < 3; ++state_index) {
            value += incoming[state_index] * lep[lbase + state_index * 16 + pindex];
          }
          dlp[pindex * H + h] += value;
        }
        for (int pindex = 0; pindex < 4; ++pindex) {
          float value = 0.0f;
          for (int state_index = 0; state_index < 3; ++state_index) {
            value += incoming[state_index] * gep[gbase + state_index * 4 + pindex];
          }
          dgp[pindex] += value;
        }

        // Advance the full hybrid trace exactly once, after contraction.
        OnlineDual budget_d = online_basis(b0, 0);
        OnlineDual energy_d = online_basis(e0, 1);
        OnlineDual saturation_d = online_basis(c0, 2);
        OnlineDual local[16] = {
          online_basis(Wgbp[h], 3), online_basis(betap[h], 4),
          online_basis(logkp[h], 5), online_basis(logrp[h], 6),
          online_basis(ksp[h], 7), online_basis(kyp[h], 8),
          online_basis(kbp[h], 9), online_basis(knp[h], 10),
          online_basis(kbiasp[h], 11), online_basis(rsp[h], 12),
          online_basis(ryp[h], 13), online_basis(rbp[h], 14),
          online_basis(rnp[h], 15), online_basis(rbiasp[h], 16),
          online_basis(kcp[h], 17), online_basis(rcp[h], 18)
        };
        OnlineDual globals[4] = {
          online_basis(alpha_logit, 19), online_basis(lambda_log, 20),
          online_basis(decay_logit, 21), online_basis(suppress_logit, 22)
        };
        OnlineDual y_d = online_constant(y);
        for (int d = 0; d < 23; ++d) y_d.derivative[d] = budget_d.derivative[d] * (1.0f - y * y);
        OnlineDual ay_d = online_constant(ay);
        OnlineDual gate_pre_d = online_constant(gate_pre);
        for (int d = 0; d < 23; ++d) {
          ay_d.derivative[d] = Ap[h * H + h] * y_d.derivative[d];
          gate_pre_d.derivative[d] = Ugp[h * H + h] * y_d.derivative[d] + local[0].derivative[d];
        }
        OnlineDual gate_d = online_hardsigmoid(gate_pre_d);
        OnlineDual novelty_d = online_tanh(online_abs(online_sub(ay_d, budget_d)));
        OnlineDual decay_d = online_sigmoid(globals[2]);
        OnlineDual stress_d = online_mul(novelty_d, online_sub(online_constant(1.0f), energy_d));
        OnlineDual cpost_d = online_add(
            online_mul(decay_d, saturation_d),
            online_mul(online_sub(online_constant(1.0f), decay_d), stress_d));
        OnlineDual suppress_d = online_sigmoid(globals[3]);
        OnlineDual novel_eff_d = online_mul(
            novelty_d,
            online_sub(online_constant(1.0f), online_mul(suppress_d, cpost_d)));
        OnlineDual ksignal_d = online_add(
            online_add(online_add(online_mul(local[4], gate_d), online_mul(local[5], y_d)),
                       online_add(online_mul(local[6], online_tanh(budget_d)), online_mul(local[7], novel_eff_d))),
            online_add(local[8], online_mul(local[14], cpost_d)));
        OnlineDual rsignal_d = online_add(
            online_add(online_add(online_mul(local[9], gate_d), online_mul(local[10], y_d)),
                       online_add(online_mul(local[11], online_tanh(budget_d)), online_mul(local[12], novel_eff_d))),
            online_add(local[13], online_mul(local[15], cpost_d)));
        OnlineDual write_cost_d = online_mul(
            online_softplus(local[2]), online_exp(online_mul(online_constant(0.5f), online_tanh(ksignal_d))));
        OnlineDual recovery_d = online_clamp01(online_mul(
            online_sigmoid(local[3]), online_exp(online_mul(online_constant(0.5f), online_tanh(rsignal_d)))));
        OnlineDual plasticity_d = online_mul(gate_d, energy_d);
        OnlineDual alpha_d = online_sigmoid(globals[0]);
        OnlineDual lambda_d = online_softplus(globals[1]);
        OnlineDual next_budget_d = online_add(
            online_add(online_mul(online_sub(online_constant(1.0f), alpha_d), budget_d), local[1]),
            online_mul(lambda_d, online_mul(plasticity_d, ay_d)));
        OnlineDual next_energy_d = online_clamp01(online_sub(
            online_add(energy_d, online_mul(recovery_d, online_sub(online_constant(1.0f), energy_d))),
            online_mul(write_cost_d, plasticity_d)));
        const OnlineDual states[3] = {next_budget_d, next_energy_d, cpost_d};
        float jacobian[3][3];
        for (int out_state = 0; out_state < 3; ++out_state) {
          for (int in_state = 0; in_state < 3; ++in_state) {
            jacobian[out_state][in_state] = states[out_state].derivative[in_state];
          }
        }
        float old_local[3][16]; float old_global[3][4];
        for (int si = 0; si < 3; ++si) {
          for (int pi = 0; pi < 16; ++pi) old_local[si][pi] = lep[lbase + si * 16 + pi];
          for (int pi = 0; pi < 4; ++pi) old_global[si][pi] = gep[gbase + si * 4 + pi];
        }
        for (int out_state = 0; out_state < 3; ++out_state) {
          for (int pi = 0; pi < 16; ++pi) {
            float value = states[out_state].derivative[3 + pi];
            for (int in_state = 0; in_state < 3; ++in_state) value += jacobian[out_state][in_state] * old_local[in_state][pi];
            lep[lbase + out_state * 16 + pi] = value;
          }
          for (int pi = 0; pi < 4; ++pi) {
            float value = states[out_state].derivative[19 + pi];
            for (int in_state = 0; in_state < 3; ++in_state) value += jacobian[out_state][in_state] * old_global[in_state][pi];
            gep[gbase + out_state * 4 + pi] = value;
          }
        }
        jp[off] = next_budget_d.derivative[0];
      }
    }
  }

  // E[t+1]=j[t]*E[t]+c[t] outer x[t]. Contract the affine recurrence
  // across the chunk using small scalar scans and BLAS. A zero j is valid:
  // no division by a cumulative product occurs anywhere.
  auto response = torch::empty({B, H}, opts);
  auto multiplier = torch::empty({B, H}, opts);
  auto future = torch::empty({B, T, H}, opts);
  auto suffix = torch::empty({B, T, H}, opts);
  float* responsep = response.data_ptr<float>();
  float* multiplierp = multiplier.data_ptr<float>();
  float* futurep = future.data_ptr<float>();
  float* suffixp = suffix.data_ptr<float>();
  for (int64_t bi = 0; bi < B; ++bi) {
    for (int64_t h = 0; h < H; ++h) {
      float carry = 0.0f, product = 1.0f;
      for (int64_t ti = T; ti-- > 0;) {
        const int64_t off = (bi * T + ti) * H + h;
        futurep[off] = carry;
        suffixp[off] = product;
        carry = ibp[off] + jp[off] * carry;
        product = jp[off] * product;
      }
      responsep[bi * H + h] = carry;
      multiplierp[bi * H + h] = product;
    }
  }
  auto slope = ((gate_pre_tape > -3.0) & (gate_pre_tape < 3.0)).to(torch::kFloat32) / 6.0;
  auto common = lam * epre;
  auto coefficient_g = common * ay_tape * slope;
  auto coefficient_a = common * gate_tensor;
  auto coefficient_w = common * (1.0f - y_tape.square()) *
      (ay_tape * slope * Ug.diagonal() + gate_tensor * A.diagonal());
  auto contract = [&](const torch::Tensor& initial, const torch::Tensor& coefficient,
                      const torch::Tensor& values, torch::Tensor& gradient) {
    auto final = torch::baddbmm(initial * multiplier.unsqueeze(-1),
                               (coefficient * suffix).transpose(1, 2), values);
    gradient.add_(torch::bmm(response.transpose(0, 1).unsqueeze(1),
                            initial.transpose(0, 1)).squeeze(1));
    gradient.addmm_((coefficient * future).reshape({B * T, H}).t(),
                    values.reshape({B * T, values.size(2)}));
    return final;
  };
  next_w_elig = contract(w_elig, coefficient_w, inputs, dW);
  next_a_elig = contract(a_elig, coefficient_a, y_tape, dA);
  next_wg_elig = contract(wg_elig, coefficient_g, inputs, dWg);
  next_ug_elig = contract(ug_elig, coefficient_g, y_tape, dUg);

  return {
      next_w_elig, next_a_elig, next_wg_elig, next_ug_elig, next_local_elig,
      next_global_elig, grad_input, dW, dA, dWg, dUg, dWg_bias + dlocal[0],
      dlocal[1], dglobal[0].reshape_as(logit_alpha), dglobal[1].reshape_as(log_lambda),
      dlocal[2], dlocal[3], dlocal[4], dlocal[5], dlocal[6], dlocal[7], dlocal[8],
      dlocal[9], dlocal[10], dlocal[11], dlocal[12], dlocal[13],
      dglobal[2].reshape_as(logit_c_decay), dlocal[14], dlocal[15],
      dglobal[3].reshape_as(logit_c_suppress)};
}

torch::Tensor permanent_delta_scan_cpu(
    torch::Tensor key, torch::Tensor value, torch::Tensor beta) {
  key = key.contiguous(); value = value.contiguous(); beta = beta.contiguous();
  const int64_t B = key.size(0), T = key.size(1), H = key.size(2);
  auto opts = key.options().dtype(torch::kFloat32);
  torch::Tensor output = torch::empty({B, T, H}, opts);
  const float* kp = key.data_ptr<float>();
  const float* vp = value.data_ptr<float>();
  const float* bp = beta.data_ptr<float>();
  float* op = output.data_ptr<float>();

  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> state(H * H, 0.0f);
    std::vector<float> recon(H, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
      const int64_t off = (bi * T + t) * H;
      for (int64_t row = 0; row < H; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < H; ++col) {
          sum += state[row * H + col] * kp[off + col];
        }
        recon[row] = sum;
        op[off + row] = sum;
      }
      const float bt = bp[bi * T + t];
      for (int64_t row = 0; row < H; ++row) {
        const float scaled_error = bt * (vp[off + row] - recon[row]);
        for (int64_t col = 0; col < H; ++col) {
          state[row * H + col] += scaled_error * kp[off + col];
        }
      }
    }
  }
  return output;
}

std::vector<torch::Tensor> permanent_delta_scan_bwd_cpu(
    torch::Tensor key, torch::Tensor value, torch::Tensor beta, torch::Tensor grad_output) {
  key = key.contiguous(); value = value.contiguous(); beta = beta.contiguous();
  grad_output = grad_output.contiguous();
  const int64_t B = key.size(0), T = key.size(1), H = key.size(2);
  auto opts = key.options().dtype(torch::kFloat32);
  torch::Tensor dkey = torch::zeros({B, T, H}, opts);
  torch::Tensor dvalue = torch::zeros({B, T, H}, opts);
  torch::Tensor dbeta = torch::zeros({B, T}, opts);
  const float* kp = key.data_ptr<float>();
  const float* vp = value.data_ptr<float>();
  const float* bp = beta.data_ptr<float>();
  const float* gp = grad_output.data_ptr<float>();
  float* dkp = dkey.data_ptr<float>();
  float* dvp = dvalue.data_ptr<float>();
  float* dbp = dbeta.data_ptr<float>();

  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    std::vector<float> state(H * H, 0.0f);
    std::vector<float> state_tape(T * H * H, 0.0f);
    std::vector<float> recon_tape(T * H, 0.0f);
    for (int64_t t = 0; t < T; ++t) {
      std::copy(state.begin(), state.end(), state_tape.begin() + t * H * H);
      const int64_t off = (bi * T + t) * H;
      for (int64_t row = 0; row < H; ++row) {
        float recon = 0.0f;
        for (int64_t col = 0; col < H; ++col) {
          recon += state[row * H + col] * kp[off + col];
        }
        recon_tape[t * H + row] = recon;
      }
      const float bt = bp[bi * T + t];
      for (int64_t row = 0; row < H; ++row) {
        const float scaled_error = bt * (vp[off + row] - recon_tape[t * H + row]);
        for (int64_t col = 0; col < H; ++col) {
          state[row * H + col] += scaled_error * kp[off + col];
        }
      }
    }

    std::vector<float> adjoint(H * H, 0.0f);
    std::vector<float> ak(H, 0.0f);
    for (int64_t t = T - 1; t >= 0; --t) {
      const int64_t off = (bi * T + t) * H;
      const float* state_i = state_tape.data() + t * H * H;
      const float bt = bp[bi * T + t];
      for (int64_t row = 0; row < H; ++row) {
        float sum = 0.0f;
        for (int64_t col = 0; col < H; ++col) {
          sum += adjoint[row * H + col] * kp[off + col];
        }
        ak[row] = sum;
        dvp[off + row] = bt * sum;
        dbp[bi * T + t] += sum * (vp[off + row] - recon_tape[t * H + row]);
      }
      for (int64_t col = 0; col < H; ++col) {
        float grad = 0.0f;
        for (int64_t row = 0; row < H; ++row) {
          grad += state_i[row * H + col] * gp[off + row];
          grad += bt * adjoint[row * H + col] * vp[off + row];
          grad -= bt * state_i[row * H + col] * ak[row];
          grad -= bt * adjoint[row * H + col] * recon_tape[t * H + row];
        }
        dkp[off + col] = grad;
      }
      for (int64_t row = 0; row < H; ++row) {
        for (int64_t col = 0; col < H; ++col) {
          adjoint[row * H + col] =
              gp[off + row] * kp[off + col]
              + adjoint[row * H + col]
              - bt * ak[row] * kp[off + col];
        }
      }
    }
  }
  return {dkey, dvalue, dbeta};
}

torch::Tensor local_field_gather_cpu(torch::Tensor inputs, torch::Tensor patch) {
  inputs = inputs.contiguous(); patch = patch.contiguous();
  const int64_t B = inputs.size(0), N = inputs.size(1), D = inputs.size(2);
  const int64_t K = patch.size(1);
  torch::Tensor output = torch::empty({B, N, K, D}, inputs.options().dtype(torch::kFloat32));
  const float* xp = inputs.data_ptr<float>();
  const int64_t* pp = patch.data_ptr<int64_t>();
  float* op = output.data_ptr<float>();
  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    for (int64_t cell = 0; cell < N; ++cell) {
      for (int64_t neighbor = 0; neighbor < K; ++neighbor) {
        const int64_t source = pp[cell * K + neighbor];
        for (int64_t dim = 0; dim < D; ++dim) {
          op[((bi * N + cell) * K + neighbor) * D + dim] =
              xp[(bi * N + source) * D + dim];
        }
      }
    }
  }
  return output;
}

torch::Tensor local_field_gather_bwd_cpu(
    torch::Tensor grad_output, torch::Tensor patch, int64_t cells) {
  grad_output = grad_output.contiguous(); patch = patch.contiguous();
  const int64_t B = grad_output.size(0), N = grad_output.size(1);
  const int64_t K = grad_output.size(2), D = grad_output.size(3);
  torch::Tensor grad_inputs = torch::zeros({B, cells, D}, grad_output.options().dtype(torch::kFloat32));
  const float* gp = grad_output.data_ptr<float>();
  const int64_t* pp = patch.data_ptr<int64_t>();
  float* xp = grad_inputs.data_ptr<float>();
  #pragma omp parallel for schedule(static) if (B > 1)
  for (int64_t bi = 0; bi < B; ++bi) {
    for (int64_t cell = 0; cell < N; ++cell) {
      for (int64_t neighbor = 0; neighbor < K; ++neighbor) {
        const int64_t source = pp[cell * K + neighbor];
        for (int64_t dim = 0; dim < D; ++dim) {
          xp[(bi * cells + source) * D + dim] +=
              gp[((bi * N + cell) * K + neighbor) * D + dim];
        }
      }
    }
  }
  return grad_inputs;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("dabsn_core_scan_cpu", &dabsn_core_scan_cpu,
        "Fused DABSN core scan (CPU/OpenMP, fp32 forward)");
  m.def("three_way_read_cpu", &three_way_read_cpu,
        "Fused DABSN three-way admitted read (CPU/OpenMP, fp32 forward)");
  m.def("three_way_read_bwd", &three_way_read_bwd,
        "Fused DABSN three-way admitted read backward (CPU/OpenMP, fp32)");
  m.def("linrec_forward_cpu", &linrec_forward_cpu,
        "Fused diagonal linear recurrence (CPU/OpenMP, fp32 forward)");
  m.def("linrec_backward_cpu", &linrec_backward_cpu,
        "Fused diagonal linear recurrence backward (CPU/OpenMP, fp32)");
  m.def("dabsn_core_scan_fwd_train", &dabsn_core_scan_fwd_train,
        "DABSN core scan forward-train with state trajectory");
  m.def("dabsn_core_scan_bwd", &dabsn_core_scan_bwd,
        "DABSN core reverse-time backward");
  m.def("dabsn_core_online_replay_cpu", &dabsn_core_online_replay_cpu,
        "DABSN core forward-time online temporal-credit replay (CPU/OpenMP, fp32)");
  m.def("permanent_delta_scan_cpu", &permanent_delta_scan_cpu,
        "DABSN permanent delta scan (CPU/OpenMP)");
  m.def("permanent_delta_scan_bwd_cpu", &permanent_delta_scan_bwd_cpu,
        "DABSN permanent delta scan backward (CPU/OpenMP)");
  m.def("local_field_gather_cpu", &local_field_gather_cpu,
        "DABSN local-field gather (CPU/OpenMP)");
  m.def("local_field_gather_bwd_cpu", &local_field_gather_bwd_cpu,
        "DABSN local-field gather backward (CPU/OpenMP)");
}
