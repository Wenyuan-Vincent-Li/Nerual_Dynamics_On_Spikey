import pyNN.brian as sim
sim.setup()

cell = sim.Population(1, sim.HH_cond_exp())
cell.record('v')
sim.run(100)
data = cell.get_data()
sim.end()