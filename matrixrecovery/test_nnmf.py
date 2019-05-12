# Ensure same random initial Y, rather than generate new one
# when executing this cell.

from myutils3_v2 import * 
import cvxpy as cvx

m = 128; n = 128; k = 5;
#A = rand(k,m)'*rand(n,k)';
A = ra.rand(m,k) @ ra.rand(k,n)

Y_init = ra.rand(m,k)
Y = Y_init 

print('Y initial:')
print(Y_init)

# Perform alternating minimization.
tt = tic()
MAX_ITERS = 100
residual = np.zeros(MAX_ITERS)
for iter_num in range(1,1+MAX_ITERS):
    # At the beginning of an iteration, X and Y are NumPy
    # array types, NOT CVXPY variables.

    # For odd iterations, treat Y constant, optimize over X.
    if iter_num % 2 == 1:
        X = cvx.Variable((k, n))
        constraint = [X >= 0]
    # For even iterations, treat X constant, optimize over Y.
    else:
        Y = cvx.Variable((m, k))
        constraint = [Y >= 0]

    # Solve the problem.
    obj = cvx.Minimize(cvx.norm(A - Y*X, 'fro'))
    prob = cvx.Problem(obj, constraint)
#    prob.solve(solver=cvx.ECOS)
    prob.solve()

    if prob.status != cvx.OPTIMAL:
        raise Exception("Solver did not converge!")

    print('Iteration {}, residual norm {}'.format(iter_num, prob.value))
    residual[iter_num-1] = prob.value

    # Convert variable to NumPy array constant for next iteration.
    if iter_num % 2 == 1:
        X = X.value
#         print('X:')
#         print(X)
    else:
        Y = Y.value
#         print('Y:')
#         print(Y)

elapsed = toc(tt)
printExpr('elapsed');

#
# Plot residuals.
#

import matplotlib.pyplot as plt
plt.ion()

# Show plot inline in ipython.
#%matplotlib inline

# Set plot properties.
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
font = {'family' : 'normal',
        'weight' : 'normal',
        'size'   : 16}
plt.rc('font', **font)

# Create the plot.
plt.plot(residual)
plt.xlabel('Iteration Number')
plt.ylabel('Residual Norm')
plt.show()

#
# Print results.
#
print('Original matrix:')
print(A)
print('Left factor Y:')
print(Y)
print('Right factor X:')
print(X)
print('Residual A - Y * X:')
print(A - Y @ X)
print('Residual after {} iterations: {}'.format(iter_num, prob.value))

