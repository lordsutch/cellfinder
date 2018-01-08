#!/usr/bin/env python3

SHARED_TOWERS = {}

# S.C. Georgia/E.C. Alabama
for i in range(0x03A42E, 0x03A437):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A42E+0x03A594-i:06X}')]

for i in range(0x03A439, 0x03A44B):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A439+0x03A58B-i:06X}')]

for i in range(0x03A466, 0x03A47F):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A550+0x03A478-i:06X}')]

# for i in range(0x03A47A, 0x03A47F):
#     SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
#         (310, 120, f'{0x03A47A+0x03A54E-i:06X}')]

for i in range(0x03A481, 0x03A489):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A481+0x03A548-i:06X}')]

for i in range(0x03A48A, 0x03A4A6):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A48A+0x03A540-i:06X}')]
    
#for i in range(0x03A49B, 0x03A4A6):
#    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
#        (310, 120, f'{0x03A49B+0x03A52F-i:06X}')]

for i in range(0x03A4A9, 0x03A4AC):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A4A9+0x03A522-i:06X}')]

for i in range(0x03A4B1, 0x03A4B8):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A4B1+0x03A51A-i:06X}')]

# Exceptions
SHARED_TOWERS[(310, 120, '03A42C')] = [(310, 120, '03A595')]
SHARED_TOWERS[(310, 120, '03A463')] = [(310, 120, '03A564')]

SHARED_TOWERS[(310, 120, '03A489')] = [(310, 120, '03A9B2')]
SHARED_TOWERS[(310, 120, '03A492')] = [(310, 120, '03A538')]
SHARED_TOWERS[(310, 120, '03A4A9')] = [(310, 120, '03A522')]
SHARED_TOWERS[(310, 120, '03A4AA')] = [(310, 120, '03A520')]

SHARED_TOWERS[(310, 120, '03A4C4')] = [(310, 120, '03A9B6')]
SHARED_TOWERS[(310, 120, '03A4A8')] = [(310, 120, '03A9B7')]
SHARED_TOWERS[(310, 120, '03A47E')] = [(310, 120, '03A9B8')]

SHARED_TOWERS[(310, 120, '03A4CA')] = [(310, 120, '0CA99A')]
SHARED_TOWERS[(310, 120, '03A4B8')] = [(310, 120, '0CA9AA')]

SHARED_TOWERS[(310, 120, '03400B')] = [(310, 120, '034936')]

# Maybe incorrect?
SHARED_TOWERS[(310, 120, '03400C')] = [(310, 120, '034956')]

SHARED_TOWERS[(310, 120, '044003')] = [(310, 120, '0444E3')]

# Memphis/N. Mississippi
for i in range(0x050113, 0x050126):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x050113+0x0508B0-i:06X}')]

for i in range(0x050125, 0x050129):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x050125+0x05089F-i:06X}')]

for i in range(0x050131, 0x050135):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x050131+0x050893-i:06X}')]

for i in range(0x050092, 0x050099):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x050922+0x050096-i:06X}')]

#SHARED_TOWERS[(310, 120, '050128')] = [(310, 120, '05089C')]
SHARED_TOWERS[(310, 120, '050124')] = [(310, 120, '0509AC')]

#SHARED_TOWERS[(310, 120, '050134')] = [(310, 120, '050890')]


#print(SHARED_TOWERS)
