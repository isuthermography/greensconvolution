# -*- coding: mbcs -*-
#
# Abaqus/CAE Release 6.14-2 replay file
# Internal Version: 2014_08_22-08.53.04 134497
# Run by sdh4 on Mon Oct  3 16:56:37 2016
#

# from driverUtils import executeOnCaeGraphicsStartup
# executeOnCaeGraphicsStartup()
#: Executing "onCaeGraphicsStartup()" in the site directory ...
from abaqus import *
from abaqusConstants import *
session.Viewport(name='Viewport: 1', origin=(0.0, 0.0), width=256.666687011719, 
    height=31.0)
session.viewports['Viewport: 1'].makeCurrent()
session.viewports['Viewport: 1'].maximize()
from caeModules import *
from driverUtils import executeOnCaeStartup
executeOnCaeStartup()
session.viewports['Viewport: 1'].partDisplay.geometryOptions.setValues(
    referenceRepresentation=ON)
Mdb()
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
execfile('/home/sdh4/genabqscript.py', __main__.__dict__)
#: A new model database has been created.
#: The model "Model-1" has been created.
session.viewports['Viewport: 1'].setValues(displayedObject=None)
#: The interaction property "CohesiveInteraction" has been created.
#: The interaction property "ContactInteraction" has been created.
#* IOError: /home/usr_local_src_el7/greensconvolution/demos/CBlock.sat: No such 
#* file or directory
#* File "/home/sdh4/genabqscript.py", line 257, in <module>
#*     acisgeom=abq.mdb.openAcis(acisfile,scaleFromFile=abqC.ON)
session.viewports['Viewport: 1'].setValues(displayedObject=None)
