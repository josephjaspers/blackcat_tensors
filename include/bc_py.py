import cppyy 


print("Attempting to import BlackCat_Tensors.h", cppyy.cppdef('#include "BlackCat_Tensors.h"')) 

matrix = cppyy.gbl.BC.Matrix[float]
vector = cppyy.gbl.BC.Vector[float]
cube  = cppyy.gbl.BC.Cube[float]

named_types = [vector, matrix, cube] 


for type in named_types: 

    def __repr__(self):
        return self.to_string() 

    setattr(type, '__repr__', __repr__)



a = matrix(3,3)
a.randomize(0, 1);
a.print()

print('a + a / a')
a = a + a / a
print(a) 

