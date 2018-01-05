#!/usr/bin/env python3

SHARED_TOWERS = {}

for i in range(0x03A466, 0x03A478):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A550+0x03A478-i:06X}')]

for i in range(0x03A47A, 0x03A47F):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A47A+0x03A54E-i:06X}')]

for i in range(0x03A481, 0x03A489):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A481+0x03A548-i:06X}')]

for i in range(0x03A48A, 0x03A492):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A48A+0x03A540-i:06X}')]
    
for i in range(0x03A49B, 0x03A4A6):
    SHARED_TOWERS[(310, 120, f'{i:06X}')] = [
        (310, 120, f'{0x03A49B+0x03A52F-i:06X}')]

# Exceptions
SHARED_TOWERS[(310, 120, '03A489')] = [(310, 120, '03A9B2')]
SHARED_TOWERS[(310, 120, '03A492')] = [(310, 120, '03A538')]
SHARED_TOWERS[(310, 120, '03A4A8')] = [(310, 120, '03A9B7')]
SHARED_TOWERS[(310, 120, '03A4A9')] = [(310, 120, '03A522')]
SHARED_TOWERS[(310, 120, '03A4AA')] = [(310, 120, '03A520')]
SHARED_TOWERS[(310, 120, '03A4C4')] = [(310, 120, '03A9B6')]
SHARED_TOWERS[(310, 120, '03A4CA')] = [(310, 120, '0CA99A')]

#print(SHARED_TOWERS)
