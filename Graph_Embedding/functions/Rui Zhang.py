# Reconstruct adjacency matrix by Rui Zhang
def Reconstruct(self, beta):
    # X : n*d
    n, d = self.X.size()
    A = (self.X).mm(self.X.T)
    A[A < 1e-4] = 0
    F = torch.sigmoid(A)
    S = torch.zeros(n, n)
    E = L2_distance_2(self.X, self.X)
    A_alpha = (F - beta * E)
    for i2 in range(n):
        tran = EProjSimplex_new(A_alpha[:, i2:i2 + 1], 1)
        S[:, i2:i2 + 1] = tran
    S = (S + S.T) / 2
    return S


def EProjSimplex_new(v, k=1):
    ft = 1
    n = v.shape[0]
    v0 = v - torch.mean(v) + k / n
    vmin = torch.min(v0)
    v1_0 = torch.zeros(n, 1)
    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 1e-4:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = sum(posidx.float())
            g = -npos
            f = sum(v1[posidx]) - k
            lambda_m = lambda_m - f / (g + 1e-6)
            ft = ft + 1
            v1_0 = v1_0.type(v1.type())
            if ft > 100:
                x = torch.max(v1, v1_0)
                break
        x = torch.max(v1, v1_0)
    else:
        x = v0
    return x
