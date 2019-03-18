#encoding: utf-8

print "Hello PYTHON World!"

#data definition
num    = 100
str    = "Pefect World of Jasper"
list   = ["Wash Mask", "TFLearning", "Wash My Own", "Grapefruit/Orange"]

print "All the data we defined:"
print "num: %s" %num
#'+' only can be used for concatenate between strings
print "str: " + str
print "list: %s" %list

for i in range(0,len(list)):
	print "list[%d] =" %i, list[i]

#do cut op to list
#[[start:end):step]
slice01 = list[0:3]
slice02 = list[:]
slice03 = list[2:3]
slice04 = list[1:2:1]
slice05 = list[:1:-1]
print "slice01: %s" %slice01
print "slice02: %s" %slice02
print "slice03:", slice03
print "slice04:", slice04
print "slice05:", slice05

#mod/del/ins
del list[3]
list.insert(3, "Grapefruit")
list.insert(4, "Orange")
list[4] = "GoldenOrange"

print "After op, list: %s" %list

#tuple -- const list
tuple01 = (1,) #if only 1 element, add the comma
tuple02 = ("China", "US", "UK", "Russia")
tuple03 = ("Name", 1, 3.1415926)

print "tuple01: ", tuple01
print "tuple02: ", tuple02
print "tuple03: ", tuple03

#dictionary
tony_file =  {"Name": "Tony",   "Age": 18, "Height": 180, "Sex": "Male"}
lucy_file =  {"Name": "Lucy",   "Aage": 22, "Height": 172, "Sex": "Female"}
jim_file =   {"Name": "Jim",    "Age": 21, "Height": 168, "Sex": "Male"}
emily_file = {"Name": "Emily",  "Age": 20, "Height": 171, "Sex": "Female"}

stu_file = [tony_file, lucy_file, jim_file, emily_file];

#Modify dict
stu_file[2]["Age"] = 23
del stu_file[1]["Aage"]
stu_file[1]["Age"] = 22

lucy_file["Height"] = 178

for i in stu_file:
	print i


while True:
	num = input("Please input your score:")
	if num > 100 or num < 0:
		print "Invalid number, please re-input"
	elif num == 0:
		break
	else:
		if num < 60:
			print "Sorry, you failed this test"
		else:
			print "Congratulations, you passed"

print "See you PYTHON world."