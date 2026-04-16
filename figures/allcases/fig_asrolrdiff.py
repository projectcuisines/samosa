#
# Comparison for SAMOSA Case 4: Full parameter space overview
#
import netCDF4

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

#--------------------------------------------------------------------
# Model Data

cases = np.array( range(16) ) + 1
flux1  = np.array( [ 500, 1900, 2400, 1200, 1500, 2100, 1600, 800, 1100, 400, 900, 1500, 1600, 900, 600, 1400 ] )

# ExoPlaSim
#diff_exoplasim = np.array( [ -0.6, -4.2, -0.4, 0.0, -0.3, -1.6, -0.5, -1.1, 0.0, -0.2, 0.0, -1.1, -0.2, -0.2, -0.1, -1.4 ] )
diff_exoplasim = np.array( [ -0.6, -4.2, -0.4, 0.0, -0.3, -1.6, -0.5, -1.1, 0.0, -0.2, 0.0, -1.1, -0.2, -0.2, -0.1, -1.4 ] ) / flux1
offset_exoplasim = 0

# ExoCAM
olr_exocam = np.array( [ 93.3625, 0, 0, 207.6944, 0, 0, 0, 159.8870, 181.0581, 70.4075, 176.9891, 307.1630, 0, 161.1979, 121.3727, 291.4003 ] )
#asr_exocam = np.array( [ ] )
#diff_exocam = np.array( [ -2.4540, 0, 0, -0.6811, 0, 0, 0, -1.1715, -0.8482, -2.1087, -1.5808, -0.7615, 0, -1.1086, -1.8869, -0.2712 ] )
diff_exocam = np.array( [ -2.4540, 0, 0, -0.6811, 0, 0, 0, -1.1715, -0.8482, -2.1087, -1.5808, -0.7615, 0, -1.1086, -1.8869, -0.2712 ] ) / flux1
offset_exocam = -0.1

# ROCKE-3D
olr_r3d  = np.array( [ 96.86504, 0, 0, 205.78123, 225.66759, 0, 223.26585, 156.65143, 171.65938, 85.158714, 160.64339, 296.16342, 236.52412, 157.14787, 116.68313, 305.1014 ] )
asr_r3d  = np.array( [ 96.203354, 0, 0, 205.45311, 225.65686, 0, 222.91484, 155.82481, 171.62772, 78.8421, 160.48454, 307.84683, 236.78061, 157.38396, 116.67793, 305.46756 ] )
#olr_r3d  = np.array( [ 96.86504, 406.96237, 0, 205.78123, 225.66759, 412.56, 223.26585, 156.65143, 171.65938, 85.158714, 160.64339, 296.16342, 236.52412, 157.14787, 116.68313, 305.1014 ] )
#asr_r3d  = np.array( [ 96.203354, 449.43787, 0, 205.45311, 225.65686, 497.65445, 222.91484, 155.82481, 171.62772, 78.8421, 160.48454, 307.84683, 236.78061, 157.38396, 116.67793, 305.46756 ] )
#diff_r3d = olr_r3d - asr_r3d
diff_r3d = ( olr_r3d - asr_r3d ) / flux1
offset_r3d = -0.2

# PlaHab
olr_plahab = np.array( [ 100.8768, 0, 0, 203.1932, 249.1617, 0, 258.1238, 144.8315, 193.1640, 89.45836, 81.11828, 239.9648, 259.1602, 161.2469, 119.9889, 224.1680 ] )
asr_plahab = np.array( [ 101.1182, 0, 0, 200.9125, 251.1883, 0, 260.4074, 140.7578, 190.2533, 99.11335, 81.67249, 241.8415, 260.5934, 154.9702, 120.0981, 225.6254 ] )
#diff_plahab = olr_plahab - asr_plahab
diff_plahab = ( olr_plahab - asr_plahab ) / flux1
offset_plahab = 0.1

# LFRic
case4 = 4
olr_lfric = np.array( [ 206.91 ] )
asr_lfric = np.array( [ 208.61 ] )
#diff_lfric = olr_lfric - asr_lfric
diff_lfric = ( olr_lfric - asr_lfric ) / flux1
offset_lfric = 0.2

# Generic PCM (no OHT)
#case4 = 4
olr_pcm = np.array( [ 258.36653113204176 ] )
asr_pcm = np.array( [ 257.670191052218 ] )
#diff_pcm = olr_pcm - asr_pcm
diff_pcm = ( olr_pcm - asr_pcm ) / flux1
offset_pcm = 0.3

# Generic PCM (with OHT)
#case4 = 4
olr_pcm_oht = np.array( [ 258.59441680450504 ] )
asr_pcm_oht = np.array( [ 256.9020589300952 ] )
#diff_pcm_oht = olr_pcm_oht - asr_pcm_oht
diff_pcm_oht = ( olr_pcm_oht - asr_pcm_oht ) / flux1
offset_pcm_oht = 0.4


#--------------------------------------------------------------------
# Set Up Figure

fig, axd = plt.subplots( 1, 1, sharex=False, figsize = (14.5,4.75) )


#--------------------------------------------------------------------
# Bar Plot

axd.bar( cases + offset_exoplasim, diff_exoplasim, width=0.1, color='#666666' )
#axd.text( 0.5, 4.3, 'ExoPlaSim', fontsize=8, color='#666666' )

f1 = axd.bar( cases + offset_exocam, diff_exocam, width=0.1, color='tab:blue' )
axd.bar_label( f1, [ '', 'X', 'X', '', 'X', 'X', 'X', '', '', '', '', '', 'X', '', '', '' ], color='tab:blue', fontsize=6 )
#axd.text( 0.5, 4.0, 'ExoCAM', fontsize=8, color='tab:blue' )

f1 = axd.bar( cases + offset_r3d, diff_r3d, width=0.1, color='tab:orange' )
axd.bar_label( f1, [ '', 'X', 'X', '', '', 'X', '', '', '', '', '', '', '', '', '', '' ], color='tab:orange', fontsize=6 )
#axd.text( 0.5, 3.7, 'ROCKE-3D', fontsize=8, color='tab:orange' )

f1 = axd.bar( cases + offset_plahab, diff_plahab, width=0.1, color='tab:purple' )
axd.bar_label( f1, [ '', 'X', 'X', '', '', 'X', '', '', '', '', '', '', '', '', '', '' ], color='tab:purple', fontsize=6 )
#axd.text( 0.5, 3.4, 'PlaHab', fontsize=8, color='tab:purple' )

f1 = axd.bar( case4 + offset_lfric, diff_lfric, width=0.1, color='k' )
#axd.text( 0.5, 3.1, 'LFRic', fontsize=8, color='k' )

f1 = axd.bar( case4 + offset_pcm, diff_pcm, width=0.1, color='tab:green' )
#axd.text( 0.5, 2.8, 'Generic PCM (no OHT)', fontsize=8, color='tab:green' )

f1 = axd.bar( case4 + offset_pcm_oht, diff_pcm_oht, width=0.1, color='darkgreen' )
#axd.text( 0.5, 2.5, 'Generic PCM (w/OHT)', fontsize=8, color='darkgreen' )


axd.set_xticks( np.arange( 1, 17, 1 ) )
#axd.set_xlabel( 'Sample Number' )

axd.set_ylabel( '(OSR - ASR) / stellar flux' )
#axd.set_ylabel( 'OSR - ASR (W m$^{-2}$)' )
#axd.set_ylim( [-5, 5] )



#--------------------------------------------------------------------
# Finalize

#fig.subplots_adjust( hspace = 0.4 )
#fig.subplots_adjust( wspace = 0.4 )

fig.savefig( "fig_asrolrdiff.png", bbox_inches='tight' )
fig.savefig( "fig_asrolrdiff.eps", bbox_inches='tight' )
#plt.show()
