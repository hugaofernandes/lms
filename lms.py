
# tarefa 21.08.2018

import sys
import numpy as np
import matplotlib.pyplot as plt


class LMS:

	def __init__(self, n, rate_learn):
		self.rate_learn = rate_learn
		self.count_train = n * rate_learn
		self.v = np.random.standard_normal(n)
		self.s = np.sin(np.arange(n) * np.pi * 0.075)
		self.v1 = []
		self.v2 = []
		self.e = []
		self.ruidos()
		self.x = self.s + self.v1
		self.memory = np.ones(10)
		self.w = np.random.random(len(self.memory))
		self.wb = 0
		
	def ruidos(self):
		v1_last = 1
		v2_last = 1
		for v in self.v:
			self.v1.append(-0.5 * v1_last + v)
			self.v2.append(0.8 * v2_last + v)
			v1_last = self.v1[-1]
			v2_last = self.v2[-1]

	def train(self):
		for x, v2 in zip(self.x, self.v2):
			memory_last = self.memory
			for i in range(len(self.memory) - 1):
				self.memory[i + 1] = memory_last[i]
			self.memory[0] = v2
			y = sum(self.memory * self.w) + self.wb
			self.e.append(x - y)
			self.w += self.memory * self.rate_learn * (x - y)
			self.wb += self.rate_learn * (x - y)

def main(argv):
	if len(argv) == 3:
		lms = LMS(int(argv[1]), float(argv[2]))
		lms.train()
		#plt.plot(lms.v, label='v(n)')
		plt.plot(lms.s, label='s(n)')
		#plt.plot(lms.x, label='x(n)')
		plt.plot(lms.e, label='e(n)')
		#plt.plot(lms.v2, label='v2(n)')
		plt.legend(loc='upper left', shadow=True, fontsize='small')
		plt.xlabel('n = ' + argv[1])
		plt.show()
	else:
		return 0

if __name__ == "__main__":
	main(sys.argv)


