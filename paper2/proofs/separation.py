"""Terminal-Jacobian separation for the B-channel eligibility, both directions.

Exact polynomial identities in c = sech(1/2)^2 and n = tanh(log(3)/2 - 1/2),
treated as independent indeterminates so the identities hold for every value.
(1) Two histories with the same full terminal state-parameter Jacobian and a
different E. (2) Two histories with the same E and a different full terminal
Jacobian. So E is not the terminal Jacobian written in another format.
"""
from fractions import Fraction as Q

from _exact import Poly


def check():
    c=Poly({(1,0):1}); n=Poly({(0,1):1}); N=1-n*n
    j=Q(11,16)
    J=[[j,Q(1,4),0],[-Q(3,64)*c,Q(1,2),0],
       [N*Q(1,16),-n*Q(1,2),Q(1,2)]]
    # Direct state-parameter injection per unit input. hardsigmoid'(0)=1/6.
    injections={"W":[Q(3,16),-Q(3,64)*c,-Q(3,16)*N],
                "Wg":[Q(1,24),-Q(1,24),0]}
    annihilating=[Poly(1),Poly(-Q(27,16)),Q(15,16)+Q(3,256)*c,
                  -Q(11,64)-Q(3,512)*c]
    cancelling=[0,0,1,-j]

    def transport(xs,u):
        h=[Poly() for _ in range(3)]; e=Poly()
        for x in xs:
            h=[sum(J[i][k]*h[k] for k in range(3))+u[i]*x for i in range(3)]
            e=j*e+u[0]*x
        return h,e

    # p(J)=0 is verified on an independent basis, not only the two injections.
    for column in range(3):
        h,_=transport(annihilating,[int(i==column) for i in range(3)])
        assert h==[0,0,0]
    for name,u in injections.items():
        h,e=transport(annihilating,u)
        assert h==[0,0,0]
        assert e==u[0]*Q(9,4096)*c
        reverse_h,reverse_e=transport(cancelling,u)
        assert reverse_e==0
        assert reverse_h[0]==Q(1,4)*u[1]
    assert transport(annihilating,injections["W"])[1]==Q(27,65536)*c
    assert transport(cancelling,injections["W"])[0][0]==-Q(3,256)*c
    return {"characteristic_polynomial_identity":"p(J)=0 exactly in Q[c,n]",
            "same_full_terminal_jacobian_different_eligibility":{
                "full_jacobian_difference":"zero for all parameters",
                "W_eligibility_difference":"27*c/65536 > 0"},
            "same_eligibility_different_full_terminal_jacobian":{
                "eligibility_difference":"zero for all trace tensors",
                "W_budget_jacobian_difference":"-3*c/256 < 0"},
            "c":"sech(1/2)^2", "n":"tanh(log(3)/2-1/2)",
            "conditions":"fixed parameters; fixed parameter-independent initial state; inputs held constant during differentiation",
            "scope":"exact polynomial identities for the stated recurrence; native/autograd check separate"}
