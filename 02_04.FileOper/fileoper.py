#encoding: utf-8

from struct import *

def ReadInputToFile(file_path):
	
	file = open(file_path, "w")

	while True:
		try:
			str = raw_input()
		except Exception as e:
			break
		file.write(str + "\n")

	file.close()


def ReadFromFile(file_path):

	file = open(file_path, "r")

	while True:
		line = file.readline()

		if not line:
			break;
		else:
			print line.strip()

	file.close()

# LittleEndian
# typedef struct {   // bmfh
#     CHAR    bfType[2];
#     DWORD   bfSize;
#     WORD    bfReserved1;
#     WORD    bfReserved2;
#     DWORD   bfOffBits;
# } BITMAPFILEHEADER;

# typedef struct {    // bmih
#     DWORD   biSize;
#     LONG    biWidth;
#     LONG    biHeight;
#     WORD    biPlanes;
#     WORD    biBitCount;
#     DWORD   biCompression;
#     DWORD   biSizeImage;
#     LONG    biXPelsPerMeter;
#     LONG    biYPelsPerMeter;
#     DWORD   biClrUsed;
#     DWORD   biClrImportant;
# } BITMAPINFOHEADER;

def BMPFileHeaderParse(bmp_fpath):

	file = open(bmp_fpath, "rb")
	bmpfileheader = file.read(0x36)

	tag, size, img_off = unpack_from("<2sL4xL", bmpfileheader)
	print "tag: %s, size %d bytes, image data start at %d." %(tag, size, img_off)

	#bisize, width, height, planes, bitcount
	bmih = unpack_from("<LllHHLLllLL", bmpfileheader, calcsize("<2sL4xL"))
	print "BMIH size: %d bytes" %bmih[0]
	print "Resolution: %d X %d" %(bmih[1], bmih[2])
	print "Planes: %d, BBP: %d" %(bmih[3], bmih[4])
	print "%s compressed." %("Be" if bmih[5] else "Not")
	print "Image data size: %d bytes." %bmih[6]

	file.close()


#ReadInputToFile("./input.txt")
#ReadFromFile("./input.txt")

BMPFileHeaderParse("./pic01.bmp")