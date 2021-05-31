from pyeda.inter import *

Bv, Ev, Bv2, Bv3, Bv4, Bv5, Bv6, Bv7, Bv8, Bv9, Bv10, Bv11, Bv12, Bv13, Bv14, Bv15, Bv16, Bv17, Bv18, Bv19, Bv20, Bv21, Bv22, Bv23, Bv24, Bv25, Bv26, Bv27, Bv28, Bv29, Bv30, Bv31, Bv32, Bv33, Bv34, Bv35, Bv36, Bv37, Bv38, Bv39, Bv40, Bv41, Bv42, Bv43, Bv44, Bv45, Bv46, Bv47, Bv48, Bv49, Bv50, Bv51, Ev2, Ev3, Sav, Bv52, Bv53, Bv54, Bv55, Bv56, Bv57, Bv58, Bv59, Bv60, Bv61, Bv62, Bv63, Bv64, Bv65, Ev4, Ev5, Ev6, Ev7, Ev8, Ev9, Ev10= map(bddvar, ['Bv','Ev','Bv2','Bv3','Bv4','Bv5','Bv6','Bv7','Bv8','Bv9','Bv10','Bv11','Bv12','Bv13','Bv14','Bv15','Bv16','Bv17','Bv18','Bv19','Bv20','Bv21','Bv22','Bv23','Bv24','Bv25','Bv26','Bv27','Bv28','Bv29','Bv30','Bv31','Bv32','Bv33','Bv34','Bv35','Bv36','Bv37','Bv38','Bv39','Bv40','Bv41','Bv42','Bv43','Bv44','Bv45','Bv46','Bv47','Bv48','Bv49','Bv50','Bv51','Ev2','Ev3','Sav','Bv52','Bv53','Bv54','Bv55','Bv56','Bv57','Bv58','Bv59','Bv60','Bv61','Bv62','Bv63','Bv64','Bv65','Ev4','Ev5','Ev6','Ev7','Ev8','Ev9','Ev10'])
D = Bv | Ev | Bv2 | Bv3 | Bv4 | Bv5 | Bv6 | Bv7 | Bv8 | Bv9 | Bv10 | Bv11 | Bv12 | Bv13 | Bv14 | Bv15 | Bv16 | Bv17 | Bv18 | Bv19 | Bv20 | Bv21 | Bv22 | Bv23 | Bv24 | Bv25 | Bv26 | Bv27 | Bv28 | Bv29 | Bv30 | Bv31 | Bv32 | Bv33 | Bv34 | Bv35 | Bv36 | Bv37 | Bv38 | Bv39 | Bv40 | Bv41 | Bv42 | Bv43 | Bv44 | Bv45 | Bv46 | Bv47 | Bv48 | Bv49 | Bv50 | Bv51 | Ev2 | (Ev3 & Sav) | Bv52 | Bv53 | Bv54 | Bv55 | Bv56 | Bv57 | Bv58 | Bv59 | Bv60 | Bv61 | Bv62 | Bv63 | Bv64 | Bv65 | Ev4 | Ev5 | Ev6 | Ev7 | Ev8 | Ev9 | Ev10

# Original decision
# Bv or (Ev /= El) or Bv2 or Bv3 or Bv4 or Bv5 or Bv6 or Bv7 or Bv8 or
# Bv9 or Bv10 or Bv11 or Bv12 or Bv13 or Bv14 or Bv15 or Bv16 or
# Bv17 or Bv18 or Bv19 or Bv20 or Bv21 or Bv22 or Bv23 or Bv24 or
# Bv25 or Bv26 or Bv27 or Bv28 or Bv29 or Bv30 or Bv31 or Bv32 or
# Bv33 or Bv34 or Bv35 or Bv36 or Bv37 or Bv38 or Bv39 or Bv40 or
# Bv41 or Bv42 or Bv43 or Bv44 or Bv45 or Bv46 or Bv47 or Bv48 or
# Bv49 or Bv50 or Bv51 or (Ev2 = El2) or
# ((Ev3 = El2) and (Sav /= Sac)) or Bv52 or Bv53 or Bv54 or Bv55 or
# Bv56 or Bv57 or Bv58 or Bv59 or Bv60 or Bv61 or Bv62 or Bv63 or
# Bv64 or Bv65 or Ev4 /= El3 or Ev5 = El4 or Ev6 = El4 or Ev7 = El4 or
# Ev8 = El4 or Ev9 = El4 or Ev10 = El4
# 
# Abstraction
# (Ev /= El) : Ev
# (Ev2 = El2) : Ev2
# ((Ev3 = El2) : Ev3
# (Sav /= Sac): Sav
# Ev4 /= El3 : Ev4
# Ev5 = El4 : Ev5
# Ev6 = El4 : Ev6
# Ev7 = El4 : Ev7
# Ev8 = El4  Ev8
# Ev9 = El4 :Ev9
# Ev10 = El4 :Ev10
