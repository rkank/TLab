import numpy as np
import random

random.seed()


class Vector:

	# define: INIT, ADD, MUL, and DIST operations depending on representation

	# representation template
	def __init_xxx(size):
		"""
		Create a new (random) hyperdimensional vector and return it
		"""
		return None

	def __add_xxx(x, y):
		"""
		Create a new hyperdimensional vector, z, that is the result of the addition (bundling) of two hyperdimensional vectors x and y
		"""
		return None

	def __mul_xxx(x, y):
		"""
		Create a new hyperdimensional vector, z, that is the result of the multoplication (binding) of two hyperdimensional vectors x and y
		"""
		return None

	def __dist_xxx(x, y):
		"""
		Return the distance between the two hyperdimensional vectors x and y
		"""
		return None

	# binary
	def __init_bsc(size):
		return np.random.randint(2, size=size)

	def __add_bsc(x, y):
		z = x + y
		z[z == 1] = np.random.randint(2, size=len(z[z == 1]))
		z[z == 2] = np.ones(len(z[z == 2]))
		return z

	def __mul_bsc(x, y):
		z = np.bitwise_xor(x, y)
		return z

	def __dist_bsc(x, y):
		z = np.bitwise_xor(x, y)
		return (np.sum(z[z == 1]) / float(len(z)))

	# biploar
	def __init_bipolar(size):
		return np.random.choice([-1.0, 1.0], size=size)

	def __add_bipolar(x, y):
		z = x + y
		z[z > 1] = 1.0
		z[z < -1] = -1.0
		z[z == 0] = np.random.choice([-1.0, 1.0], size=len(z[z == 0]))
		return z

	def __mul_bipolar(x, y):
		return x * y

	def __dist_bipolar(x, y):
		return (len(x) - np.dot(x, y)) / (2 * float(len(x)))

	# binary sparse
	def __init_bsd(size):
		# TODO make probability as a param
		# sparsity << 0.5
		sparsity = 0.2
		return np.random.choice([0, 1], size=size, p=[1 - sparsity, sparsity])

	def __add_bsd(x, y):
		# bundling in BSD is nothing but a fancy binding, role-filler scheme - use same code
		# TODO refactor to explicitly reuse the same function

		z0 = np.bitwise_or(x, y)
		# permutation factor
		k = 8

		zk = np.zeros((k, x.shape[0]), dtype=int)
		for i in range(0, k):
			zk[i] = np.random.permutation(z0)
		z = np.bitwise_or.reduce(zk)

		return np.bitwise_and(z, z0)

	def __mul_bsd(x, y):
		z0 = np.bitwise_or(x, y)
		# permutation factor
		k = 8
		# zk = np.fromfunction(lambda i, j: np.random.permutation(z0), (k, 1), dtype=int)
		# z = np.bitwise_or.reduce(zk)

		zk = np.zeros((k, x.shape[0]), dtype=int)
		for i in range(0, k):
			zk[i] = np.random.permutation(z0)
		z = np.bitwise_or.reduce(zk)

		return np.bitwise_and(z, z0)

	def __dist_bsd(x, y):
		d = 1 - np.sum(np.bitwise_and(x, y)) / np.sqrt(np.sum(x) * np.sum(y))
		return d

	# operations list
	__OPERATIONS = {
		'bsc': {
			'init': __init_bsc,
			'add': __add_bsc,
			'mul': __mul_bsc,
			'dist': __dist_bsc
		},
		'bsd': {
			'init': __init_bsd,
			'add': __add_bsd,
			'mul': __mul_bsd,
			'dist': __dist_bsd
		},
		'bipolar': {
			'init': __init_bipolar,
			'add': __add_bipolar,
			'mul': __mul_bipolar,
			'dist': __dist_bipolar
		},
	}

	# init random HDC vector
	def __init__(self, size, rep):
		self.rep = rep
		self.size = size
		self.value = self.__OPERATIONS[rep]['init'](size)

	# print vector
	def __repr__(self):
		return np.array2string(self.value)

	# print vector
	def __str__(self):
		return np.array2string(self.value)

	# addition
	def __add__(self, a):
		b = Vector(self.size, self.rep);
		b.value = self.__OPERATIONS[self.rep]['add'](self.value, a.value)
		return b

	# multiplication
	def __mul__(self, a):
		b = Vector(self.size, self.rep);
		b.value = self.__OPERATIONS[self.rep]['mul'](self.value, a.value)
		return b

	# distance
	def dist(self, a):
		return self.__OPERATIONS[self.rep]['dist'](self.value, a.value)


class Space:

	def __init__(self, size=1000, rep='bsc'):
		self.size = size
		self.rep = rep
		self.vectors = {}

	def _random_name(self):
		return ''.join(random.choice('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ') for i in range(8))

	def __repr__(self):
		return ''.join("'%s' , %s\n" % (v, self.vectors[v]) for v in self.vectors)

	def __getitem__(self, x):
		return self.vectors[x]

	def add(self, name=None):
		if name == None:
			name = self._random_name()

		v = Vector(self.size, self.rep)

		self.vectors[name] = v
		return v

	def insert(self, v, name=None):
		if name == None:
			name = self._random_name()

		self.vectors[name] = v

		return name

	def find(self, x):
		d = 1.0
		match = None

		for v in self.vectors:
			if self.vectors[v].dist(x) < d:
				match = v
				d = self.vectors[v].dist(x)

		# print d
		return match, d