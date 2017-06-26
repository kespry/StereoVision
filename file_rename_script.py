import os
camera = '{0:s}'.format(input('left or right: '))
assert(camera=='left' or camera=='right')
folder = '{0:s}'.format(input('image path :'))

"""Discover stereo photos and return them as a pairwise sorted list."""
files = os.listdir(folder)
files.sort()

for i,f in enumerate(files):
	#print(folder + '/' + camera + '-'+ str(i) + '.jpg')
	os.rename(folder+'/'+f,folder + '/' + camera + '-'+ str(i) + '.jpg')