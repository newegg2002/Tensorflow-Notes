#encoding: utf-8

class HomeAppliance(object):
	"""docstring for HomeAppliance"""
	def __init__(self, power):
		super(HomeAppliance, self).__init__()
		self.power = power

	def PowerOn(self):
		print "Powering %s On..." %self.__class__.__name__

	def PowerOff(self):
		print "Powering %s Off..." %self.__class__.__name__

class TVSet(HomeAppliance):
	"""docstring for TVSet"""
	def __init__(self, power, brand, disp_size):
		super(TVSet, self).__init__(power)
		self.brand = brand
		self.disp_size = disp_size
	
	def ChangeChannel(self, new_chan):
		print "Change to Channel %d" %new_chan

class Lamp(HomeAppliance):
	"""docstring for Lamp"""
	def __init__(self, power, color):
		super(Lamp, self).__init__(power)
		self.color = color

	def Dim(self):
		print "We're going to dim..."

washer = HomeAppliance(100)
print "The power of Washer is %d" %washer.power
washer.PowerOn()
washer.PowerOff()

mainTV = TVSet(120, "LG", 65)

print "The power of mainTV is %d" %mainTV.power
print "The brand of mainTV is", mainTV.brand
print "We got a %d\" mainTV" %mainTV.disp_size

mainTV.PowerOn()
mainTV.ChangeChannel(13)
mainTV.PowerOff()


bedroomLamp = Lamp(40, "Red")

print "The power of bedroomLamp is %d" %bedroomLamp.power
print "We have a %s bedroom lamp" %bedroomLamp.color

bedroomLamp.PowerOn()
bedroomLamp.Dim()
bedroomLamp.PowerOff()
