import sympy as sp
import numpy as np

a0,e0,i0,omega0,Omega0,nu0,a1,e1,i1,omega1,Omega1,nu1 = sp.symbols("a_0,e_0,i_0,omega_0,Omega_0,nu_0,a_1,e_1,i_1,omega_1,Omega_1,nu_1")

r = sp.symbols('r')


#Declare term elements and exponents
individual_terms = [a0,e0,sp.sin(i0),sp.cos(i0),Omega0,sp.sin(nu0),sp.cos(nu0),sp.sin(nu0),sp.cos(nu0),a1,e1,Omega1,sp.sin(nu1),sp.cos(nu1),sp.sin(nu1),sp.cos(nu1),sp.sin(i1),sp.cos(i1)]
ns = [-4,-3,-2,-1,1,2,3,4]


#expand terms to all exponents of terms
individual_terms2 = list()
for i in np.arange(len(ns)):
    for j in np.arange(len(individual_terms)):
        individual_terms2.append(individual_terms[j]**ns[i])
        #individual_terms2.append(-individual_terms[j]**ns[i])

individual_terms2

len(individual_terms2)



a0,e0,i0,W0,w0,v0,x0,y0,z0 = sp.symbols("a_0,e_0,i_0,Omega_0,omega_0,nu_0,x_0,y_0,z_0",real=True)
a1,e1,i1,W1,w1,v1,x1,y1,z1 = sp.symbols("a_1,e_1,i_1,Omega_1,omega_1,nu_1,x_1,y_1,z_1",real=True)
a,e,i,W,w,v,x,y,z = sp.symbols("a,e,i,Omega,omega,nu,x,y,z",real=True)


Ap,Bp,Cp,Dp,Ep,Fp,Gp,Hp,Ip,Jp,Kp = sp.symbols('A_p,B_p,C_p,D_p,E_p,F_p,G_p,H_p,I_p,J_p,K_p',real=True)
Aq,Bq,Cq,Dq,Eq,Fq,Gq,Hq,Iq,Jq,Kq = sp.symbols('A_q,B_q,C_q,D_q,E_q,F_q,G_q,H_q,I_q,J_q,K_q',real=True)
vq, vp, ep, eq = sp.symbols('nu_q, nu_p, e_p, e_q',real=True)

B,C,D,E = sp.symbols('B,C,D,E',real=True) #replacements for coefficients in above

#XYZ
r = a*(1-e**2)/(1+e*sp.cos(v))
X = r*(sp.cos(W)*sp.cos(w+v)-sp.sin(W)*sp.sin(w+v)*sp.cos(i))
Y = r*(sp.sin(W)*sp.cos(w+v)+sp.cos(W)*sp.sin(w+v)*sp.cos(i))
Z = r*sp.sin(i)*sp.sin(w+v)

A = sp.symbols('A',real=True,positive=True) #The replacement for a(1-e^2)
Xp = X.subs(a*(1-e**2),A)
X0 = sp.expand(X).subs(sp.sin(W)*sp.cos(i)*sp.cos(w),B).subs(sp.sin(W)*sp.sin(w)*sp.cos(i),C).subs(sp.sin(w)*sp.cos(W),D).subs(sp.cos(W)*sp.cos(w),E)


heqn = (-(Ap*Jp*sp.sin(vp)/(ep*sp.cos(vp) + 1) + Ap*Kp*sp.cos(vp)/(ep*sp.cos(vp) + 1))*(-Ap*Jp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Kp*sp.cos(vp)/(ep*sp.cos(vp) + 1) + Aq*Jq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Kq*sp.cos(vq)/(eq*sp.cos(vq) + 1)) + (-Ap*Bp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Cp*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Ap*Dp*sp.sin(vp)/(ep*sp.cos(vp) + 1) + Ap*Ep*sp.cos(vp)/(ep*sp.cos(vp) + 1))*(-Ap*Bp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Cp*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Ap*Dp*sp.sin(vp)/(ep*sp.cos(vp) + 1) + Ap*Ep*sp.cos(vp)/(ep*sp.cos(vp) + 1) + Aq*Bq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Cq*sp.cos(vq)/(eq*sp.cos(vq) + 1) + Aq*Dq*sp.sin(vq)/(eq*sp.cos(vq) + 1) - Aq*Eq*sp.cos(vq)/(eq*sp.cos(vq) + 1)) - (-Ap*Fp*sp.sin(vp)/(ep*sp.cos(vp) + 1) + Ap*Gp*sp.cos(vp)/(ep*sp.cos(vp) + 1) + Ap*Hp*sp.sin(vp)/(ep*sp.cos(vp) + 1) + Ap*Ip*sp.cos(vp)/(ep*sp.cos(vp) + 1))*(Ap*Fp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Gp*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Ap*Hp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Ip*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Aq*Fq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Gq*sp.cos(vq)/(eq*sp.cos(vq) + 1) + Aq*Hq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Iq*sp.cos(vq)/(eq*sp.cos(vq) + 1)))/sp.sqrt((-Ap*Jp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Kp*sp.cos(vp)/(ep*sp.cos(vp) + 1) + Aq*Jq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Kq*sp.cos(vq)/(eq*sp.cos(vq) + 1))**2 + (Ap*Bp*sp.sin(vp)/(ep*sp.cos(vp) + 1) + Ap*Cp*sp.cos(vp)/(ep*sp.cos(vp) + 1) + Ap*Dp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Ep*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Aq*Bq*sp.sin(vq)/(eq*sp.cos(vq) + 1) - Aq*Cq*sp.cos(vq)/(eq*sp.cos(vq) + 1) - Aq*Dq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Eq*sp.cos(vq)/(eq*sp.cos(vq) + 1))**2 + (Ap*Fp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Gp*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Ap*Hp*sp.sin(vp)/(ep*sp.cos(vp) + 1) - Ap*Ip*sp.cos(vp)/(ep*sp.cos(vp) + 1) - Aq*Fq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Gq*sp.cos(vq)/(eq*sp.cos(vq) + 1) + Aq*Hq*sp.sin(vq)/(eq*sp.cos(vq) + 1) + Aq*Iq*sp.cos(vq)/(eq*sp.cos(vq) + 1))**2)


n,d = sp.fraction(heqn)


n2 = sp.simplify(n*(ep*sp.cos(vp)+1)*(eq*sp.cos(vq)+1))**2


d2 = sp.simplify(d**2*(ep*sp.cos(vp)+1)**2*(eq*sp.cos(vq)+1)**2)


#solving for h^2
h2eqn = n2/d2

d2_expanded = sp.expand(d2)


baseterms = [sp.sin(vp),sp.cos(vp),sp.sin(vq),sp.cos(vq)]
pows = [1,2]
# baseterms2 = list()
# for i in np.arange(len(baseterms)):
#     for j in np.arange(len(pows)):
#         baseterms2.append(baseterms[i]**pows[j])


import itertools

#itertools.combinations(baseterms,2)
combs = list()
for i in np.arange(len(baseterms))+1:
    for term in itertools.combinations(baseterms,i):
        combs.append(term)

combs



combs2 = list()
for i in np.arange(len(combs)):
    powCombs = list()
    for ppow in itertools.product(pows, repeat=len(combs[i])):
        powCombs.append(ppow)
    for j in np.arange(len(powCombs)):
        tmpCombs = list()
        for k in np.arange(len(combs[i])):
            tmpCombs.append(combs[i][k]**powCombs[j][k])
        combs2.append(tmpCombs)

combs2

powCombs


#we can set the equations equal because we know h
h2 = sp.symbols('h2',real=True,positive=True)
expr = n2 - h2*d2

#Find the fundamental sin and cos terms driving the expression
expr2 = expr*(ep*sp.cos(vp)+1)**2


expr3 = expr2
for sym in [Ap,Bp,Cp,Dp,Ep,Fp,Gp,Hp,Ip,Jp,Kp,Aq,Bq,Cq,Dq,Eq,Fq,Gq,Hq,Iq,Jq,Kq,h2]:
    expr3 = expr3.subs(sym,1)

import time

start = time.time()
print("Simplifying expr3")
expr4 = sp.simplify(expr3)
stop = time.time()
print("Done Simplifying expr3 " + str(stop-start))


#divide out that expression and split up the args
expr3_args = list((expr3/(ep*sp.cos(vp)+1)**2).args)
for i in np.arange(len(expr3_args)):
    expr3_args[i] = expr3_args[i]*(ep*sp.cos(vp)+1)**2

#Expand all of the individual arguments
expr3_args_expanded = list()
for i in np.arange(len(expr3_args)):
    expr3_args_expanded.append(sp.expand(expr3_args[i]))

#Create an expression containing just the sin cos combinations
expr3_all_args = list()
for i in np.arange(len(expr3_args_expanded)):
    tmp_args = list(expr3_args_expanded[i].subs(ep,1).subs(eq,1).args)
    for tmp_arg in tmp_args:
        tmp_arg = tmp_arg.as_independent(vp,vq)[1]
        #Duplicate Check
        anyDup = False
        for j in np.arange(len(expr3_all_args)):
            if expr3_all_args[j] == tmp_arg:
                anyDup = True
                break
        if anyDup == False:
            expr3_all_args.append(tmp_arg)



print("Expanding expr4")
start = time.time()
expr4_expanded = sp.expand(expr4)
stop = time.time()
print("Done Expanding expr4 " + str(stop-start))



#Create an expression containing just the sin cos combinations
expr4_args_expanded = expr4_expanded.args
expr4_all_args = list()
for i in np.arange(len(expr4_args_expanded)):
    tmp_args = list(expr4_args_expanded[i].subs(ep,1).subs(eq,1).args)
    for tmp_arg in tmp_args:
        tmp_arg = tmp_arg.as_independent(vp,vq)[1]
        #Duplicate Check
        anyDup = False
        for j in np.arange(len(expr4_all_args)):
            if expr4_all_args[j] == tmp_arg:
                anyDup = True
                break
        if anyDup == False:
            expr4_all_args.append(tmp_arg)





from sympy.plotting import plot3d 

for i in np.arange(len(expr3_all_args)):
    plot3d(expr3_all_args[i],(vp,0,2*sp.pi),(vq,0,2*sp.pi),title=str(expr3_all_args[i]),block=False)


# #If this hangs here or does not run, then I will need to reevaluate
# print("Simplifying expr2")
# start = time.time()
# expr2_simplified = sp.simplify(expr2)
# stop = time.time()
# print("Done Simplifying expr2 " + str(stop-start))
# print(expr2_simplified)


# print("expanding expr2")
# start = time.time()
# expr2_expanded = sp.expand(expr2)
# stop = time.time()
# print("Done expanding expr2 " + str(stop-start))
# print(expr2_expanded)



terms = [sp.cos(nu_p)**2,
 sp.cos(nu_p)**4,
 sp.cos(nu_q)**2,
 sp.sin(nu_p)**2,
 sp.sin(nu_q)**2,
 sp.cos(nu_p)**3,
 sp.cos(nu_p)**2*sp.cos(nu_q)**2,
 sp.sin(nu_p)**2*sp.cos(nu_p)**2,
 sp.sin(nu_q)**2*sp.cos(nu_p)**4,
 sp.sin(nu_p)**2*sp.cos(nu_q)**2,
 sp.sin(nu_q)**2*sp.cos(nu_p)**2,
 sp.sin(nu_p)*sp.cos(nu_p)**2,
 sp.sin(nu_q)**2*sp.cos(nu_p)**3,
 sp.sin(nu_q)**2*sp.cos(nu_p),
 sp.sin(nu_p)*sp.cos(nu_p)**3,
 sp.cos(nu_p)*sp.cos(nu_q)**2,
 sp.sin(nu_p)**2*sp.cos(nu_p),
 sp.sin(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_p)*sp.cos(nu_p),
 sp.sin(nu_q)*sp.cos(nu_q),
 sp.cos(nu_p)**3*sp.cos(nu_q),
 sp.sin(nu_q)*sp.cos(nu_p)**4,
 sp.sin(nu_p)*sp.cos(nu_q)**2,
 sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_q)*sp.cos(nu_p),
 sp.sin(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)*sp.sin(nu_q),
 sp.cos(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_q)*sp.cos(nu_p)**2,
 sp.sin(nu_q)*sp.cos(nu_p)**3,
 sp.sin(nu_p)**2*sp.cos(nu_p)**2*sp.cos(nu_q)**2,
 sp.sin(nu_q)*sp.cos(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_q)*sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.cos(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_p)*sp.cos(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_p)*sp.cos(nu_p)**3*sp.cos(nu_q),
 sp.sin(nu_q)*sp.cos(nu_p)**3*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.cos(nu_p)*sp.cos(nu_q)**2,
 sp.sin(nu_p)*sp.cos(nu_p)**2*sp.cos(nu_q)**2,
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_p)**3,
 sp.sin(nu_p)*sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_q),
 sp.sin(nu_p)*sp.cos(nu_p)*sp.cos(nu_q)**2,
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_p)**2,
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_p),
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_p)**3*sp.cos(nu_q),
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_p)*sp.sin(nu_q)*sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)**4,
 sp.sin(nu_p)**3*sp.sin(nu_q),
 sp.sin(nu_p)**3*sp.cos(nu_q)**2,
 sp.sin(nu_p)**3*sp.cos(nu_q),
 sp.sin(nu_p)**3*sp.cos(nu_p),
 sp.sin(nu_p)**4*sp.cos(nu_q)**2,
 sp.sin(nu_p)**2*sp.sin(nu_q)**2,
 sp.sin(nu_p)**4*sp.cos(nu_q),
 sp.sin(nu_p)**3*sp.sin(nu_q)*sp.cos(nu_p),
 sp.sin(nu_p)**3*sp.sin(nu_q)*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.sin(nu_q)*sp.cos(nu_p)**2,
 sp.sin(nu_p)**2*sp.sin(nu_q)*sp.cos(nu_p),
 sp.sin(nu_p)*sp.sin(nu_q)**2*sp.cos(nu_p)**3,
 sp.sin(nu_p)**3*sp.cos(nu_p)*sp.cos(nu_q)**2,
 sp.sin(nu_p)**2*sp.sin(nu_q)*sp.cos(nu_q),
 sp.sin(nu_p)*sp.sin(nu_q)**2*sp.cos(nu_p),
 sp.sin(nu_p)*sp.sin(nu_q)**2*sp.cos(nu_p)**2,
 sp.sin(nu_p)**2*sp.sin(nu_q)**2*sp.cos(nu_p)**2,
 sp.sin(nu_p)**3*sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.sin(nu_q)**2*sp.cos(nu_p),
 sp.sin(nu_p)**3*sp.sin(nu_q)*sp.cos(nu_p)*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.sin(nu_q)*sp.cos(nu_p)**2*sp.cos(nu_q),
 sp.sin(nu_p)**2*sp.sin(nu_q)*sp.cos(nu_p)*sp.cos(nu_q)]


terms2 = [sp.sin(2*nu_p),
 1,
 sp.sin(nu_p + nu_q),
 sp.cos(nu_p - nu_q),
 sp.cos(nu_p)**2,
 sp.cos(nu_q)**2,
 sp.sin(nu_p)**2,
 sp.sin(nu_q)**2,
 sp.sin(nu_p + pi/4)**2,
 sp.sin(nu_q + pi/4)**2,
 sp.cos(nu_p - nu_q)**2,
 sp.cos(nu_q),
 sp.cos(2*nu_p - nu_q),
 sp.cos(nu_p)**3,
 sp.sin(nu_q),
 sp.sin(2*nu_p + nu_q),
 sp.cos(nu_p)**4,
 sp.sin(nu_p + pi/4),
 sp.sin(nu_q + pi/4),
 sp.sin(2*nu_p - nu_q),
 sp.cos(nu_p),
 sp.sin(nu_p),
 sp.sin(2*nu_p + nu_q)**2,
 sp.sin(2*nu_p - nu_q)**2,
 sp.cos(2*nu_p - nu_q)**2]



