def h(theta,x):
    return x.dot(theta)

def linear_reg_cost_function(theta,x,y,lmd):
    m=y.size
    myh=h(theta,x)
    cost = (myh - y).dot(myh - y) / (2 * m) + theta[1:].dot(theta[1:]) * (lmd / (2 * m))

    grad=(myh-y).dot(x)/m
    grad[1:]+=(lmd/m)*theta[1:]
    return cost,grad

