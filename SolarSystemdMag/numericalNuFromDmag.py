# numericalNuFromDmag


#This is a solver for nu from dmag assuming a quasi-lambert phase function
#Generate terms
lhs = 10.**(-0.4*dmag)*a**2.*(e**2. + 1.)**2./(R_p*p)

A = e**4.*sin(i)**4.*sin(w)**4./16. + e**4.*sin(i)**4.*sin(w)**2.*cos(w)**2./8. + e**4.*sin(i)**4.*cos(w)**4./16.
B = e**4.*sin(i)**3.*sin(w)**3./4. + e**4.*sin(i)**3.*sin(w)*cos(w)**2./4. + e**3.*sin(i)**4.*sin(w)**4./4. + e**3.*sin(i)**4.*sin(w)**2.*cos(w)**2./2. + e**3.*sin(i)**4.*cos(w)**4./4.
C = -e**4.*sin(i)**4.*sin(w)**2.*cos(w)**2./8. - e**4.*sin(i)**4.*cos(w)**4./8. + 3.*e**4.*sin(i)**2.*sin(w)**2./8. + e**4.*sin(i)**2.*cos(w)**2./8. + e**3.*sin(i)**3.*sin(w)**3. + e**3.*sin(i)**3.*sin(w)*cos(w)**2. + 3.*e**2.*sin(i)**4.*sin(w)**4./8. + 3.*e**2.*sin(i)**4.*sin(w)**2.*cos(w)**2./4. + 3.*e**2.*sin(i)**4.*cos(w)**4./8.
D = -e**4.*sin(i)**3.*sin(w)*cos(w)**2./4. + e**4.*sin(i)*sin(w)/4. - e**3.*sin(i)**4.*sin(w)**2.*cos(w)**2./2. - e**3.*sin(i)**4.*cos(w)**4./2. + 3*e**3.*sin(i)**2.*sin(w)**2./2. + e**3.*sin(i)**2.*cos(w)**2./2. + 3*e**2.*sin(i)**3.*sin(w)**3./2. + 3*e**2.*sin(i)**3.*sin(w)*cos(w)**2./2. + e*sin(i)**4.*sin(w)**4./4. + e*sin(i)**4.*sin(w)**2.*cos(w)**2./2. + e*sin(i)**4.*cos(w)**4./4.
E = e**4.*sin(i)**4.*cos(w)**4./16. - e**4.*sin(i)**2.*cos(w)**2./8. + e**4./16. - e**3.*sin(i)**3.*sin(w)*cos(w)**2. + e**3.*sin(i)*sin(w) - e**2.*lhs*sin(i)**2.*sin(w)**2./2. + e**2.*lhs*sin(i)**2.*cos(w)**2./2. - 3*e**2.*sin(i)**4.*sin(w)**2.*cos(w)**2./4. - 3*e**2.*sin(i)**4.*cos(w)**4./4. + 9*e**2.*sin(i)**2.*sin(w)**2./4. + 3*e**2.*sin(i)**2.*cos(w)**2./4. + e*sin(i)**3.*sin(w)**3. + e*sin(i)**3.*sin(w)*cos(w)**2. + sin(i)**4.*sin(w)**4./16. + sin(i)**4.*sin(w)**2.*cos(w)**2./8. + sin(i)**4.*cos(w)**4./16.
F = e**3.*sin(i)**4.*cos(w)**4./4. - e**3.*sin(i)**2.*cos(w)**2./2. + e**3./4. - e**2.*lhs*sin(i)*sin(w) - 3*e**2.*sin(i)**3.*sin(w)*cos(w)**2./2. + 3*e**2.*sin(i)*sin(w)/2. - e*lhs*sin(i)**2.*sin(w)**2. + e*lhs*sin(i)**2.*cos(w)**2. - e*sin(i)**4.*sin(w)**2.*cos(w)**2./2. - e*sin(i)**4.*cos(w)**4./2. + 3*e*sin(i)**2.*sin(w)**2./2. + e*sin(i)**2.*cos(w)**2./2. + sin(i)**3.*sin(w)**3./4. + sin(i)**3.*sin(w)*cos(w)**2./4.
G = -e**2.*lhs*sin(i)**2.*cos(w)**2./2. - e**2.*lhs/2. + 3*e**2.*sin(i)**4.*cos(w)**4./8. - 3*e**2.*sin(i)**2.*cos(w)**2./4. + 3*e**2./8. - 2*e*lhs*sin(i)*sin(w) - e*sin(i)**3.*sin(w)*cos(w)**2. + e*sin(i)*sin(w) - lhs*sin(i)**2.*sin(w)**2./2. + lhs*sin(i)**2.*cos(w)**2./2. - sin(i)**4.*sin(w)**2.*cos(w)**2./8. - sin(i)**4.*cos(w)**4./8. + 3*sin(i)**2.*sin(w)**2./8. + sin(i)**2.*cos(w)**2./8.
H = -e*lhs*sin(i)**2.*cos(w)**2. - e*lhs + e*sin(i)**4.*cos(w)**4./4. - e*sin(i)**2.*cos(w)**2./2. + e/4. - lhs*sin(i)*sin(w) - sin(i)**3.*sin(w)*cos(w)**2./4. + sin(i)*sin(w)/4.
I = lhs**2. - lhs*sin(i)**2.*cos(w)**2./2. - lhs/2. + sin(i)**4.*cos(w)**4./16. - sin(i)**2.*cos(w)**2./8. + 1/16.

# Coefficients of xth degree polynomial
coeffs = [-A**2. - B**2.,
 -2.*A*C - 2.*B*D,
 -2.*A*E + B**2. - 2.*B*F - C**2. - D**2.,
 -2.*A*G + 2.*B*D - 2.*B*H - 2.*C*E - 2.*D*F,
 -2.*A*I + 2.*A*lhs + 2.*B*F - 2.*C*G + D**2. - 2.*D*H - E**2. - F**2.,
 2.*B*H - 2.*C*I + 2.*C*lhs + 2.*D*F - 2.*E*G - 2.*F*H,
 2.*D*H - 2.*E*I + 2.*E*lhs + F**2. - G**2. - H**2.,
 2.*F*H - 2.*G*I + 2.*G*lhs,
 H**2. - I**2. + 2.*I*lhs - lhs**2.]

 out = np.roots(coeffs)

 