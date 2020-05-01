import numpy as np 
from scipy.stats import norm
from scipy.optimize import brentq, least_squares

class OptionPricer:
    cp:  "call or put"
    ul:  "underlying price"
    sp:  "strike price"
    dte: "days to expiry (in days)"
    ir:  "inerest rates"
    coc: "cost of carry"
    iv : "impled volatility"
    diy: "days in year"

    def __init__(self):
        return

    def european_premium(self, cp , ul,  sp,  dte,  ir,  coc,  iv , diy):
        # Convert to year format. Eg. dte = 30 days to expirty, diy = 365 days a year
        # Convert to 30 / 365
        dte = dte / diy

        # Convert call or put to lower case 
        cp = cp.lower()

        # Calculattions
        d1 = (np.log(ul / sp) + (coc + iv * iv / 2) * dte) / (iv * np.sqrt(dte))
        d2 = d1 - iv * np.sqrt(dte)

        if ("c" in cp):
            return (ul * np.exp((coc - ir) * dte) * norm.cdf(d1) - sp * np.exp(-ir * dte) * norm.cdf(d2))
        if ("p" in cp):
            return (sp * np.exp(-ir * dte) * norm.cdf(-d2) - ul * np.exp((coc - ir) * dte) * norm.cdf(-d1))
        return None

    def american_premium(self, cp, ul,  sp,  dte,  ir,  coc,  iv,  diy):
        cp = cp.lower()
        if ("c" in cp):
            return self._american_premium_call(ul, sp, dte, ir, coc, iv, diy)
        else:
            return self._american_premium_call(sp, ul, dte, ir - coc, -coc, iv, diy)


    def _american_premium_call(self, ul,  sp,  dte,  ir,  coc,  iv,  diy):

        b_inifinity = 0
        b0 = 0
        ht = 0
        i = 0
        alpha = 0
        beta = 0

        if (coc >= ir):
            return self.european_premium("Call", ul, sp, dte, ir, coc, iv, diy)
       
        dte = dte / diy
        beta = (1.0 / 2 - coc / iv ** 2) + np.sqrt((coc / iv**2 - 1.0 / 2) ** 2 + 2.0 * ir / iv **2)
        if beta > 100:
            return 0

        b_inifinity = beta / (beta - 1) * sp
        b0 = max(sp, ir / (ir - coc) * sp)
        ht = -(coc * dte + 2.0 * iv * np.sqrt(dte)) * b0 / (b_inifinity - b0)
        i = b0 + (b_inifinity - b0) * (1 - np.exp(ht))
        alpha = (i - sp) * (i ** (-beta))
        if (ul >= i):
            return (ul - sp)
  
        temp1 = alpha * (ul ** beta) - alpha * self._my_phi(ul, dte, beta, i, i, ir, coc, iv)
        temp2 = self._my_phi(ul, dte, 1, i, i, ir, coc, iv)
        temp3 = self._my_phi(ul, dte, 1, sp, i, ir, coc, iv)
        temp4 = sp * self._my_phi(ul, dte, 0, i, i, ir, coc, iv)
        temp5 = sp * self._my_phi(ul, dte, 0, sp, i, ir, coc, iv)
        return temp1 + temp2 - temp3 - temp4 + temp5

    def _my_phi(self, ul , dte , gamma , h , i , ir , coc , iv):
        myphi = 0
        mylambda = 0
        kappa = 0
        d = 0

        mylambda = (-ir + gamma * coc + 0.5 * gamma * (gamma - 1) * iv**2) * dte
        d = -(np.log(ul / h) + (coc + (gamma - 0.5) * (iv ** 2)) * dte) / (iv * np.sqrt(dte))
        kappa = 2 * coc / (iv ** 2 ) + (2 * gamma - 1)
        myphi_temp1 = (norm.cdf(d) - (i / ul) ** kappa * norm.cdf(d - 2 * np.log(i / ul) / (iv * np.sqrt(dte))))
        myphi_temp2 =  ul ** gamma
        try:
            myphi = np.exp(mylambda) * myphi_temp2 * myphi_temp1
        except:
            return np.nan
        return myphi

    def get_implied_volatility(self, cp, ul, sp, dte, ir, coc, diy, premium , is_american = True):
        my_func = self.american_premium  if is_american else self.european_premium
        def Root(SIGMA):
            term1 = my_func(cp, ul, sp, dte, ir, coc, SIGMA, diy)
            res =  term1 - premium
            return res
        results = brentq(Root, 0.0001, 3.0, disp = True)
        return results    


