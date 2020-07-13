import random, itertools, copy, analysis.utils

class UniqueFactory:
  def __init__(self, initial, randomize=False):
    self.all = [ x for x in initial ]

    if randomize != False:
      if type(randomize) == int:
        random.seed( randomize )
      random.shuffle( self.all )

    self.changing = copy.deepcopy( self.all )

  def __len__(self):
    return len(self.changing)

  def take(self):
    try:
      value = self.changing.pop(0) # Pop from front of list
    except IndexError:
      raise IndexError("No more elements in UniqueFactory")
    return value

  def give(self, x, *additional):
    all_new_values = [x, *additional]
    self.changing.extend( all_new_values ) # Push to end of list
    for k in all_new_values:
      if k not in self.all:
        self.all.append( k )

  def clone(self):
    return copy.deepcopy( self )


class RotatingFactory:
  def __init__(self, initial, randomize=False):
    self.all = [ x for x in initial ]
    self.index = 0

    if randomize != False:
      if isinstance(randomize,int):
        random.seed( randomize )
      random.shuffle( self.all )

  def __len__(self):
    return len(self.all)

  def post_increment(self):
    value = self.index
    self.index = ( self.index + 1 ) % len(self)
    return value

  def take(self):
    return_value = self.all[ self.post_increment() ]
    return return_value

  def give(self, x, *additional):
    all_new_values = [x, *additional]
    for k in all_new_values:
      if k not in self.all:
        self.all.append( k )

  def clone(self):
    return copy.deepcopy( self )


class UniqueProductFactory(UniqueFactory):
  def __init__( self, collections, offset=0, product_function=analysis.utils.shifted_product, *args, **kwargs ):
    self.all_collections = collections
    super().__init__( list(product_function( *self.all_collections )), *args, **kwargs )
class RotatingProductFactory(RotatingFactory):
  def __init__( self, collections, *args, **kwargs ):
    self.all_collections = collections
    super().__init__( list(product_function( *self.all_collections )), *args, **kwargs )
