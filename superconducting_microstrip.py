#!/usr/bin/env python
# -*- coding: utf-8 -*-

#Based on an article of an internship report
#"Implementation of a superconducting microstrip line model in APLAC"

#Copyright (C) 2015 Dumur Étienne

#This program is free software; you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation; either version 2 of the License, or
#(at your option) any later version.

#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
#GNU General Public License for more details.

#You should have received a copy of the GNU General Public License along
#with this program; if not, write to the Free Software Foundation, Inc.,
#51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import scipy.constants as cst
from scipy.optimize import fsolve


class SuperconductingMicrostrip:



    def __init__(self, epsilon_r = 11.68, w = 100e-9, h = 500e-9,
                       t_s = 50e-9, t_g = 100e-9, lambda_0 = 16e-9):
        """Class for the calculation of a microstrip line model
            
            Input:
                - epsilon_r (float) : Relative permitivity of the substrate in farad per meter.
                - w         (float) : Width of the central line in meter.
                - h         (float) : Height of the substrate.
                - t_s       (float) : Thickness of the metal layer in meter.
                - t_g       (float) : Thickness of the ground layer in meter.
                - lambda_0  (float) : Effective penetration depth in meter.
            
            
        """
        
        self.epsilon_r = epsilon_r
        
        self.w = w # Microstrip width
        self.h = h # Substrate height
        
        self.t_s = t_s # Microstrip thickness
        
        self.t_g = t_g # Ground thickness
        
        self.lambda_0 = lambda_0 # Penetration length at 0 K




    def _t_h(self):
        """Return the normalised strip thickness"""
        
        return self.t_s/self.h




    def _u(self):
        """Return the aspect ratio"""
        
        return self.w/self.h




    def _u_eff(self):
        """Return the effective aspect ratio"""
        
        return self._u()\
               + self._t_h()\
                 *(1. + 1./np.cosh(np.sqrt(self.epsilon_r - 1.)))\
                 *np.log(1. + 4.*np.exp(1)\
                 /self._t_h()*np.tanh(np.sqrt(6.517*self._u()))**2.)\
                 /2./cst.pi




    def _p(self):
        """Return intermediate of calculation"""
        
        return 2.*(1. + self._t_h())**2. - 1.\
               + np.sqrt((2.*(1. + self._t_h())**2. - 1.)**2. -1.)




    def epsilon_eff(self, f):
        """Return the effective permitivity as function of frequency f"""
        
        a = 1. + np.log((self._u_eff()**4. + (self._u_eff()/52.)**2.)\
                 /(self._u_eff()**4. + 0.432))/49.\
               + np.log(1. + (self._u_eff()/18.1)**3.)/18.7
        
        b = 0.564*((self.epsilon_r - 0.9)/(self.epsilon_r + 3.))**0.053
        
        e_0 = (self.epsilon_r + 1.)/2.\
              + (self.epsilon_r - 1.)/2.*(1. + 10./self._u_eff())**(-a*b)
        
        f_n = f*self.h/1e6
        
        p4 = 1. + 2.751*(1. - np.exp(-(self.epsilon_r/15.916)**8.))
        p3 = 0.0363*np.exp(-4.6*self._u_eff())*(1. - np.exp(-(f_n/38.7)**4.97))
        p2 = 0.33622*(1. - np.exp(-0.03442*self.epsilon_r))
        p1 = 0.27488 + self._u_eff()*(0.6315 + 0.525/(1. + 0.0157*f_n)**20.)\
                     - 0.065683*np.exp(-8.7513*self._u_eff())
        pf = p1*p2*((0.1844 + p3*p4)*f_n)**1.5763
        
        return self.epsilon_r - (self.epsilon_r - e_0)/(1. + pf)




    def _r_b(self):
        
        q = cst.pi/2.*self._u_eff()*np.sqrt(self._p())\
            + (self._p() + 1.)/2.*(1. + np.log(4./(np.sqrt(self._p()) + 1.)))\
            - np.sqrt(self._p())*np.log(np.sqrt(self._p()) + 1.)\
            - (np.sqrt(self._p()) - 1.)**2./2.*np.log(np.sqrt(self._p()) - 1.)
        
        r_b0 = q + (self._p() + 1.)/2.*np.log(max(self._p(), q))
        
        if self._u_eff() >= 5.:
            return r_b0
        else:
            return r_b0 - np.sqrt((r_b0 - 1.)*(r_b0 - self._p()))\
                       + (self._p() + 1.)\
                         *np.arctanh(np.sqrt((r_b0 - self._p())\
                         /(r_b0 - .1)))\
                       - 2.*np.sqrt(self._p())\
                           *np.arctanh(np.sqrt((r_b0 - self._p())\
                           /self._p()/(r_b0 - .1)))\
                       + cst.pi*self._u_eff()*np.sqrt(self._p())/2.



    def _k(self):
        
        r_a = np.exp(- 1. - cst.pi*self._u_eff()/2. + np.log(4.*self._p())\
                     - (np.sqrt(self._p()) + 1.)**2./(2.*np.sqrt(self._p())\
                                            *np.log(np.sqrt(self._p()) + 1.))
                     + (np.sqrt(self._p()) - 1.)**2./(2.*np.sqrt(self._p())\
                                            *np.log(np.sqrt(self._p()) - 1.)))
        
        return 2.*(np.log(2.*self._r_b()) - np.log(r_a))/cst.pi/self._u_eff()



    def C(self, f):
        """Return the capacitance per unit length in farad per meter"""
        
        return self.epsilon_eff(f)*cst.epsilon_0*self.w*self._k()/self.h





    def L(self):
        """Return the inductance per unit length in henry per meter"""
        
        return cst.mu_0/self.w/self._k()\
               *(self.h + self.lambda_0*(1./np.tanh(self.t_s/self.lambda_0)\
                 + 2.*np.sqrt(self._p())/self._r_b()\
                   /np.sinh(self.t_s/self.lambda_0)\
                 + 1./np.tanh(self.t_g/self.lambda_0)))





    def Z(self, f):
        """Return the characteristic impedance in ohm"""
        
        return np.sqrt(self.L()/self.C(f))





    def f0(self, l):
        """Return the resonance frequency of a halwave resonator in Hz"""
        
        def func(f):
            
            f_0 = cst.pi/l/np.sqrt(self.C(f)*self.L())/2./cst.pi
            
            return f_0 - f
        
        return fsolve(func, 10e9)[0]




    def get_resonator_length(self, f_0):
        """Find the resonator length leading to a resonance frequency f0"""
        
        def func(l, f_0):
            
            return self.f0(l) - f_0
        
        return fsolve(func, 1e-3, args=(f_0))[0]
