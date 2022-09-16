class LinearSchedule:
    """Set up a linear hyperparameter schedule (e.g. for dqn's epsilon parameter)"""

    def __init__(self, burnin: int, initial_value: float, final_value: float, decay_time: int):
        self.initial_value = initial_value
        self.final_value = final_value
        self.decay_time = decay_time
        self.burnin = burnin

    def __call__(self, frame: int) -> float:
        if frame < self.burnin:
            return self.initial_value
        else:
            frame = frame - self.burnin

        slope = (self.final_value - self.initial_value) / self.decay_time
        if self.final_value < self.initial_value:
            return max(slope * frame + self.initial_value, self.final_value)
        else:
            return min(slope * frame + self.initial_value, self.final_value)
