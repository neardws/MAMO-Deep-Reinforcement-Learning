import sys
sys.path.append(r"/home/neardws/Documents/AoV-Journal-Algorithm/")
from absl import app
from Experiment import run_mad3pg, run_d3pg

if __name__ == '__main__':
    app.run(run_mad3pg.main)
