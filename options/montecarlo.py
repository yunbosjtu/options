import numpy as np 
from options.numerical import OptionPricer

class OptionPricerMC(OptionPricer):

    cp:  "call or put"
    ul:  "underlying price"
    sp:  "strike price"
    dte: "days to expiry (in days)"
    ir:  "inerest rates"
    coc: "cost of carry"
    iv : "impled volatility"
    diy: "days in year"

    def __init__(self):
        super(OptionPricerMC, self).__init__()
        return

    def geometric_brownian_motion(self, ul, iv ,  ir , dte , diy , nobs) -> "ndarray. row = t, columns = paths":
        dt = 1 / diy

        # Initialize the array
        S = np.zeros((dte + 1, nobs))    
        S[0] = ul

        # Discount 
        df = np.exp(-ir * dt)

        for t in range(1, dte + 1):
            S[t] = S[t - 1] * np.exp((ir - 0.5 * iv ** 2) * dt + iv * np.sqrt(dt) * np.random.standard_normal(nobs))
        return S

    def american_premium_mc(self, cp , ul,  sp,  dte,  ir,  coc,  iv , diy , nobs , underlying_process = None):

        if underlying_process is None:
            underlying_process = self.geometric_brownian_motion( ul = ul , iv = iv , ir = ir , dte = dte , diy = diy , nobs = nobs)

        cp = cp.lower()
        payoff = np.maximum(underlying_process - sp, 0) if "c" in cp else np.maximum(sp - underlying_process, 0)

            
        # LSM algorithm
        V = np.copy(payoff)

        # Discount 
        df = np.exp(-ir * 1/diy)
        for t in range(dte - 1, 0, -1):
            reg = np.polyfit(underlying_process[t], V[t + 1] * df, 7) # Using 7 degree of freedom to find the continuation value for Option Price
            C = np.polyval(reg, underlying_process[t])
            
            # If Continuation value > payoff then take discounted value of pay off at T+1 else take pay off value at time T
            V[t] = np.where(C > payoff[t], V[t + 1] * df, payoff[t]) 
        
        # calculate MCS estimator
        optionPrice = df * 1 / nobs * np.sum(V[1])
        
        return optionPrice