"""
pipeline driver for runing within Jupyter notebook
"""
from tlpipe.pipeline.pipeline import *

import tlpipe

class run_pipeline(object):

    def __init__(self, params):

        #pipe_logging = 'info'
        pipe_logging = 'critical'
        pipe_copy = False
    
        self._p = locals()
        self._p.update(params())
        self._p_default = self._p.copy()

        #print "feedback = %d"%_p['pipe_feedback']

    @property
    def param(self):
        return self._p

    def add(self, params):

        self._p.update(params())

    def run(self):

        Manager(self._p, feedback=self._p['pipe_feedback']).run()
        self.clr()

    def clr(self): 
        #self._p.update(self._p_default)
        del self._p
        self._p = self._p_default.copy()

    def __call__(self):

        self.run()

    def __setitem__(self, key, value):

        self._p.update( {key : value} )

if __name__ == "__main__":
    import doctest
    doctest.testmod()
