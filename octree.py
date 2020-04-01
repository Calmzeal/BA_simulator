
class Quadtree:
    def __init__(self,node_list):
        if node_list is None:
            node_list = [(1,3),(2,4),(3,9)]
        self.node_list = node_list
        self.update_tree()

    def update_tree(self):

        alpha = 1./len(self.node_list)
        self.bare_centre = (
            sum(map(lambda x:x[0], self.node_list))*alpha,
            sum(map(lambda x:x[1], self.node_list))*alpha)

        self.relative_location_list = \
            list(map(lambda x:(x[0]-self.bare_centre[0],x[1]-self.bare_centre[1]), self.node_list))

        self.radius = max(
            max(map(lambda x:x[0], self.relative_location_list)),
            -min(map(lambda x:x[0], self.relative_location_list)),
            max(map(lambda x:x[1], self.relative_location_list)),
            -min(map(lambda x:x[1], self.relative_location_list)))

        self.build_tree()

    def build_tree(self):
        for location in self.relative_location_list:
            x,y = location




