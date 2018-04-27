#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from math import pow
from numpy import inf
from numpy.linalg import norm
from pyomo.contrib.trustregion.param import *
from pyomo.contrib.trustregion.filterMethod import FilterElement, Filter
from pyomo.contrib.trustregion.helper import cloneXYZ, packXYZ, minIgnoreNone, maxIgnoreNone
from pyomo.contrib.trustregion.Logger import IterLog, Logger
from pyomo.contrib.trustregion.PyomoInterface import PyomoInterface, ROMType

from pyomo.util.config import ConfigBlock


def TRF(m,eflist):
    """
    The main function of the Trust Region Filter algorithm

    m is a PyomoModel containing ExternalFunction() objects
    Model requirements: m is a nonlinear program, with exactly one active objective function.

    eflist is a list of ExternalFunction objects that should be treated with the
    trust region


    Return:
    model is solved, variables are at optimal solution or other exit condition.
    model is left in reformulated form, with some new variables introduced
    in a block named "tR" TODO: reverse the transformation.
    """

    print('dmv')
    CONFIG.display('all')

    logger = Logger()
    filteR = Filter()
    problem = PyomoInterface(m,eflist)
    x, y, z = problem.getInitialValue()


    trustRadius = CONFIG.get('trust radius').value()
    sampleRadius = CONFIG.get('sample radius').value()
    sampleregion_yn = CONFIG.get('sample region').value()
    


    iteration = -1

    romParam, yr = problem.buildROM(x, sampleRadius)
    #y = yr
    rebuildROM = False
    xk, yk, zk = cloneXYZ(x, y, z)
    chik = 1e8
    thetak = norm(yr - yk,1)

    objk = problem.evaluateObj(x, y, z)

    while True:
        if iteration >= 0:
            logger.printIteration(iteration)
            #print(xk)
        # increment iteration counter
        iteration = iteration + 1
        if iteration > CONFIG.get('max it').value():
            print("EXIT: Maxmium iterations\n")
            break

        ######  Why is this here ###########
        if iteration == 1:
            sampleregion_yn = False
        ################################

        # Keep Sample Region within Trust Region
        if trustRadius < sampleRadius:
            sampleRadius = max(CONFIG.get('sample radius adjust').value() * 
                                   trustRadius, 
                               CONFIG.get('delta min').value())
            rebuildROM = True

        #Generate a RM r_k (x) that is κ-fully linear on sigma k
        if(rebuildROM):
            #TODO: Ask Jonathan what variable 1e-3 should be
            if trustRadius < 1e-3:
                problem.romtype = ROMType.linear
            else:
                problem.romtype = CONFIG.get('reduced model type').value()

            romParam, yr = problem.buildROM(x, sampleRadius)
            #print(romParam)
            #print(sampleRadius)



        # Criticality Check
        if iteration > 0:
            flag, chik = problem.criticalityCheck(x, y, z, romParam)
            if (not flag):
                raise Exception("Criticality Check fails!\n")

        # Save the iteration information to the logger
        logger.newIter(iteration,xk,yk,zk,thetak,objk,chik)

        # Check for Termination
        if (thetak < CONFIG.get('ep i').value() and
            chik < CONFIG.get('ep chi').value() and 
            sampleRadius < CONFIG.get('ep delta').value()):
            print("EXIT: OPTIMAL SOLUTION FOUND")
            break

        # If trust region very small and no progress is being made, terminate
        # The following condition must hold for two consecutive iterations.
        if (trustRadius <= CONFIG.get('delta min').value() and 
            thetak < CONFIG.get('ep i').value()):
            if subopt_flag:
                print("EXIT: FEASIBLE SOLUTION FOUND ")
                break
            else:
                subopt_flag = True
        else:
            # This condition holds for iteration 0, which will declare the
            # boolean subopt_flag
            subopt_flag = False

        # New criticality phase
        if not sampleregion_yn:
            sampleRadius = trustRadius/2.0
            if sampleRadius > chik * CONFIG.get('criticality check').value():
                sampleRadius = sampleRadius/10.0
            trustRadius = sampleRadius*2
        else:
            sampleRadius = max(
                min(sampleRadius, 
                    chik * CONFIG.get('criticality check').value()),
                CONFIG.get('delta min').value())

        logger.setCurIter(trustRadius=trustRadius,sampleRadius=sampleRadius)

        # Compatibility Check (Definition 2)
        # radius=max(kappa_delta*trustRadius*min(1,kappa_mu*trustRadius**mu), 
        #            delta_min)
        radius = max(CONFIG.get('kappa delta').value() * 
                     trustRadius *
                     min(1, 
                         CONFIG.get('kappa mu').value() * 
                         pow(trustRadius,CONFIG.get('mu').value())),
                     CONFIG.get('delta min').value())

        try:
            flag, obj = problem.compatibilityCheck(
                x, y, z, xk, yk, zk, romParam, radius, 
                CONFIG.get('compatibility penalty').value())
        except:
            print("Compatibility check failed, unknown error")
            raise

        if not flag:
            raise Exception("Compatibility check fails!\n")


        theNorm = norm(x - xk, 2)**2 + norm(z - zk, 2)**2
        if (obj - CONFIG.get('compatibility penalty').value() * theNorm > 
            CONFIG.get('ep compatibility').value()):
            # Restoration stepNorm
            yr = problem.evaluateDx(x)
            theta = norm(yr - y, 1)
            
            logger.iterlog.restoration = True
            
            fe = FilterElement(
                objk - CONFIG.get('gamma f').value() * thetak,
                (1 - CONFIG.get('gamma theta').value()) * thetak)
            filteR.addToFilter(fe)
            
            rhok = 1 - ((theta - CONFIG.get('ep i').value()) /
                        max(thetak, CONFIG.get('ep i').value()))
            if rhok < CONFIG.get('eta1').value():
                trustRadius = max(CONFIG.get('gamma c').value() * trustRadius,
                                  CONFIG.get('delta min').value())
            elif rhok >= CONFIG.get('eta2').value():
                trustRadius = min(CONFIG.get('gamma e').value() * trustRadius,
                                  CONFIG.get('radius max').value())

            obj = problem.evaluateObj(x, y, z)

            stepNorm = norm(packXYZ(x-xk,y-yk,z-zk),inf)
            logger.setCurIter(stepNorm=stepNorm)

        else:

            # Solve TRSP_k
            flag, obj = problem.TRSPk(x, y, z, xk, yk, zk, 
                                      romParam, trustRadius)
            if not flag:
                raise Exception("TRSPk fails!\n")

            # Filter
            yr = problem.evaluateDx(x)

            stepNorm = norm(packXYZ(x-xk,y-yk,z-zk),inf)
            logger.setCurIter(stepNorm=stepNorm)

            theta = norm(yr - y, 1)
            fe = FilterElement(obj, theta)

            if not filteR.checkAcceptable(fe) and iteration>0:
                logger.iterlog.rejected = True
                trustRadius = max(CONFIG.get('gamma c').value() * stepNorm, 
                                  CONFIG.get('delta min').value())
                rebuildROM = False
                x, y, z = cloneXYZ(xk, yk, zk)
                continue

            # Switching Condition and Trust Region update
            if (((objk - obj) >= CONFIG.get('kappa theta').value() * 
                 pow(thetak, CONFIG.get('gamma s').value()))
                and 
                (thetak < CONFIG.get('theta min').value())):
                logger.iterlog.fStep = True

                trustRadius = min(
                    max(CONFIG.get('gamma e').value() * stepNorm, trustRadius),
                    CONFIG.get('radius max').value())

            else:
                logger.iterlog.thetaStep = True

                fe = FilterElement(
                    obj - CONFIG.get('gamma f').value() * theta,
                    (1 - CONFIG.get('gamma theta').value()) * theta)
                filteR.addToFilter(fe)

                # Calculate rho for theta step trust region update
                rhok = 1 - ((theta - CONFIG.get('ep i').value()) / 
                            max(thetak, CONFIG.get('ep i').value()))
                if rhok < CONFIG.get('eta1').value():
                    trustRadius = max(CONFIG.get('gamma c').value() * stepNorm,
                                      CONFIG.get('delta min').value())
                elif rhok >= CONFIG.get('eta2').value():
                    trustRadius = min(
                        max(CONFIG.get('gamma e').value() * stepNorm, 
                            trustRadius), 
                        CONFIG.get('radius max').value())



        # Accept step
        rebuildROM = True
        xk, yk, zk = cloneXYZ(x, y, z)
        thetak = theta
        objk = obj


    logger.printVectors()
#    problem.reverseTransform()
