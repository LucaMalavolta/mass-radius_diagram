from subroutines.constants import *

#M_jup_to_ear = 317.828133
#R_jup_to_ear = 11.209

M_val = np.asarray([6.6, 1.1, 1.1], dtype=np.double)
R_val = np.asarray([2.29, 0.23, 0.23], dtype=np.double)
print M_val/M_jup_to_ear, R_val/R_jup_to_ear


M_val = np.asarray([3.1, 1.2, 1.3], dtype=np.double)
R_val = np.asarray([1.77, 0.18, 0.18], dtype=np.double)
print M_val/M_jup_to_ear, R_val/R_jup_to_ear

M_val = np.asarray([2.7, 0.8, 1.2], dtype=np.double)
R_val = np.asarray([1.65, 0.17, 0.17], dtype=np.double)

print M_val/M_jup_to_ear, R_val/R_jup_to_ear
