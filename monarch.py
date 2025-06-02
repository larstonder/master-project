from monarch.rendering.omnire import OmniRe
from monarch.simulator.nuplan import nuPlan
from monarch.planner.simple_planner import SimplePlanner
from monarch.evaluation.simple_evaluator import SimpleEvaluator

def main():
    renderer = OmniRe()
    simulator = nuPlan()
    planner = SimplePlanner()
    evaluator = SimpleEvaluator()

if __name__ == "__main__":
    main()
