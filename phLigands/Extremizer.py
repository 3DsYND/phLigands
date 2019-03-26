import logging

import numpy as np


class Extremizer():
    iter_field = 40
    logger = logging.getLogger("Extremizer")
    
    def __init__(self, iter_field=40):
        self.iter_field = iter_field
        
    def extremum(self, metric, model, dataX, dataY, def_params, var_params):
        prev_params = start
        while(True):
            iter_params = prev_params.copy()
            for param in iter_params.keys():
                errors = np.empty(field[param].shape[0])
                errors[:] = np.inf
                for i, param_val in np.ndenumerate(field[param]):
                    iter_params[param] = param_val
                    errors[i] = metric.score(model, iter_params, dataX, dataY)
                    self.logger.debug(f"ParamField: {iter_params} : {errors[i]}")
                best_val = metric.extremum(filter(lambda x: not np.isnan(x), errors) )
                iter_params[param] = field[param][np.where(errors == best_val)[0]][0]
            self.logger.info(f"Iter: {iter_params}")
            if iter_params == prev_params:
                return iter_params
            prev_params = iter_params
            
class CoordEx(Extremizer):
    pass

class Gradient():
    alpha = None
    precision = None
    max_iterations = 1000
    logger = logging.getLogger("Extremizer")
    
    def __init__(self, alpha=1, precision=0.005):
        self.alpha = alpha * 0.02
        self.precision = precision
        
    def _grad_dict(self, metric, model, dataX, dataY, def_params, var_params):
        step = 0.00001
        grad = var_params.copy()
        for param in var_params:
            params_m = def_params.copy()
            params_m[param] -= step / 2
            params_p = def_params.copy()
            params_p[param] += step / 2
            
            par_diff = (metric.score(model, params_p, dataX, dataY) - metric.score(model, params_m, dataX, dataY)) / step
            grad[param] = par_diff
        return grad
        
    def extremum(self, metric, model, dataX, dataY, def_params, var_params, debug=False):
        params_arr = list()
        iteration = 0
        prev_params = def_params
        grad_zero = self._grad_dict(metric, model, dataX, dataY, def_params, var_params)
        while True:
            dlt = list()
            params = prev_params.copy()
            grad = self._grad_dict(metric, model, dataX, dataY, params, var_params)
            for param in grad:
                ## * (var_params[param][1] - var_params[param][0]) / abs(grad_zero[param])
                grad_step = self.alpha * grad[param] * (var_params[param][1] - var_params[param][0]) / abs(grad_zero[param])
                params[param] -= grad_step
                dlt.append(abs(params[param] - prev_params[param]))
            prev_params = params
            if debug: params_arr.append(params)
            iteration += 1
            
            if max(dlt) < self.precision or iteration > self.max_iterations:
                if debug:
                    ret = dict()
                    ret["params"] = params
                    ret["iterations"] = iteration
                    ret["params_arr"] = params_arr
                    return ret
                else:
                    return params
            
