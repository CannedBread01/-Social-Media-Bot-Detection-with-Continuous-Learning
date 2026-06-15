"""
ContinualExperimentManager
==========================
This is Task 2: the continual learning experiment setup.

It manages:
  - The performance matrix (rows = learning step, columns = task being evaluated)
  - Checking if the clusterer correctly found the new class (Task 3 lite)
  - Expanding the classifier when a new class arrives
  - Tracking metrics: Average Accuracy and Forgetting Measure

How the performance matrix works:
    After training on task T, we evaluate on ALL test splits seen so far (0..T).
    We write those accuracy scores into row T of the matrix.

    Example with 4 tasks:

              Task0  Task1  Task2  Task3
    Step 0  [ 0.95,  0.00,  0.00,  0.00 ]   ← trained on task0, tested on task0 only
    Step 1  [ 0.88,  0.91,  0.00,  0.00 ]   ← trained on task1, tested on task0 AND task1
    Step 2  [ 0.82,  0.87,  0.93,  0.00 ]   ← trained on task2, tested on all three
    Step 3  [ 0.79,  0.84,  0.90,  0.95 ]   ← trained on task3, tested on all four

    The diagonal = accuracy right after learning that task (plasticity).
    Off-diagonal drops = forgetting (stability).
"""

import numpy as np


class ContinualExperimentManager:
    def __init__(self, num_tasks, use_intervention_override=True):
        """
        Args:
            num_tasks (int):
                How many sequential datasets/bot-types you'll train on in total.
                This determines the size of the performance matrix.

            use_intervention_override (bool):
                Max's requested toggle flag (Task 3 lite).
                - True  → if the clusterer fails to detect the new class,
                          we manually force the classifier to expand anyway.
                - False → if the clusterer fails, we do nothing and let the
                          model try to cope (will likely hurt performance).
        """
        # The 2D performance matrix.
        # performance_matrix[step][task] = accuracy of the model at 'step' on 'task'.
        self.performance_matrix = np.full((num_tasks, num_tasks), np.nan)

        self.num_tasks = num_tasks
        self.current_step = 0          # which learning step we're on (0-indexed)
        self.use_intervention_override = use_intervention_override

        # History of per-task metrics, stored as plain dicts for easy inspection
        self.history = []

        #Performance Matrix for baseline accuracy
        self.baseline_accuracy = [ 
            0.379 , # Cresci17
            0.596, # Cresci18 
            0.973 # Caverlee11
            0.443 #Twi20
            0.860 # Twi22
        ]

    # ------------------------------------------------------------------
    # STEP 1: Clusterer check + classifier expansion  (Task 3 lite)
    # ------------------------------------------------------------------

    def check_clusterer_and_expand(self, clusterer_labels, ground_truth_labels):
        clusterer_labels = np.array(clusterer_labels)
        valid_clusters = clusterer_labels[clusterer_labels != -1]
        new_class_detected = len(np.unique(valid_clusters)) > 0

        if new_class_detected:
            print(f"✓ Clusterer detected new cluster(s).")
            return True
        else:
            print("⚠ Clusterer did NOT detect any new class.")
            if self.use_intervention_override:
                print(f"  → [INTERVENTION ON] Will force expansion.")
                return True
            else:
                print("  → [INTERVENTION OFF] No expansion.")
                return False

    # ------------------------------------------------------------------
    # STEP 2: Recording results into the matrix
    # ------------------------------------------------------------------

    def record_accuracy(self, task_index, accuracy):
        """
        Record the accuracy score for a specific past task at the current step.

        Call this once per past task after each training phase.

        Args:
            task_index (int): Which task's test split we just evaluated (0-indexed).
            accuracy (float): Accuracy score (0.0 to 1.0).
        """
        if task_index > self.current_step:
            raise ValueError(
                f"task_index ({task_index}) can't be greater than current_step ({self.current_step}). "
                f"You can only evaluate tasks you've already trained on."
            )
        self.performance_matrix[self.current_step, task_index] = accuracy

    # ------------------------------------------------------------------
    # STEP 3: Advancing to the next task
    # ------------------------------------------------------------------

    def advance_to_next_task(self):
        """
        Call this after you've finished training AND evaluating all test splits
        for the current step. It saves a snapshot and moves the step counter forward.
        """
        # Save a snapshot of metrics for this step
        snapshot = {
            "step": self.current_step,
            "avg_accuracy": self.calculate_average_accuracy(),
            "forgetting": self.calculate_forgetting_measure(),
            "row": self.performance_matrix[self.current_step, :self.current_step + 1].tolist(),
        }
        self.history.append(snapshot)

        print(f"\n--- Step {self.current_step} complete ---")
        print(f"    Avg Accuracy (this step): {snapshot['avg_accuracy']:.4f}")
        print(f"    Forgetting Measure:       {snapshot['forgetting']:.4f}")
        print(f"    Row scores: {[f'{x:.3f}' for x in snapshot['row']]}")

        self.current_step += 1
        print(f"\n=== Now starting Step {self.current_step} ===\n")

    # ------------------------------------------------------------------
    # METRICS
    # ------------------------------------------------------------------

    def calculate_average_accuracy(self):
        """
        Average Accuracy (AA) — measures how well the model currently performs
        across all tasks it has been trained on so far. Higher = better plasticity.

        Formula: mean of performance_matrix[current_step, 0..current_step]
        """
        row = self.performance_matrix[self.current_step, :self.current_step + 1]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            return 0.0
        return float(np.mean(valid))

    def calculate_forgetting_measure(self):
        """
        Forgetting Measure (FM) — measures how much performance on old tasks has
        degraded since they were first learned. Higher = more catastrophic forgetting.

        Formula: for each past task j, take (peak accuracy ever achieved on j)
                 minus (current accuracy on j), then average across all past tasks.

        Returns 0.0 if we're still on the first task (nothing to forget yet).
        """
        if self.current_step < 1:
            return 0.0

        forgetting_scores = []
        for j in range(self.current_step):   # for each past task
            col = self.performance_matrix[:self.current_step + 1, j]
            valid = col[~np.isnan(col)]
            if len(valid) < 2:
                continue
            peak = float(np.max(valid))
            current = float(self.performance_matrix[self.current_step, j])
            if not np.isnan(current):
                forgetting_scores.append(peak - current)

        if not forgetting_scores:
            return 0.0
        return float(np.mean(forgetting_scores))

    def calculate_forward_transfer(self):
        """
        needs the baseline accuracies.

        calculates the forward transfer(fwt), which measures how much earlier information help with the accuracy on the new task

        for fwt > 0: forward transfer is positive and earlier learned information helps on the current task
        for fwt < 0: forward transfer is negative and thus inhibits the model on the new task
        for fwt = 0: foward transfer has no impact

        Formula: FWT = (1 / (T*(T-1)/2)) * sum_{i<j} (R[i,j] - baseline[j])
        """
        total = 0.0
        count = 0

        for learned_task in range(self.num_tasks):

            for future_task in range(learned_task + 1, self.num_tasks):

                value = self.performance_matrix[
                    learned_task,
                    future_task,
                ]

                if np.isnan(value):
                    continue
                
                total += (
                    value
                    - self.baseline_accuracy[future_task]
                )
                count += 1

        return total / count if count > 0 else 0.0
    
    def intransigence(self):
        """
        needs baseline_accuracy and gets outputed in the end.

        calculates intransigence, which measures how much we accuracy we lose by learning tasks incrementally and not jointly.
        
        The difference between us and our baseline model

        Formula: I = (1/T) * sum{j} (baseline_accuracy_matrix[j] - our_accuracy_matrix[j]), with T = len(baseline_accuracy_matrix)

        :return: Average loss by learning tasks incrementally
        """
        CL_accuracy = self.performance_matrix[-1, :]

        total = 0.0
        count = 0
        
        for j in range(self.num_tasks):

            if np.isnan(CL_accuracy[j]):
                continue

            total += (self.baseline_accuracy[j] - CL_accuracy[j])
            count += 1

        return total / count if count > 0 else 0.0
        
    def compute_average_incremental_accuracy(self):
        """
        gets outputed once in the end.

        Describes how well the model performs throughout the whole learning process.

        Formula:Formula: AIA = (1/T) * sum{i=0..T-1} ( (1/(i+1)) * sum{j=0..i} our_accuracy_matrix[i,j] ), with T = len(our_accuracy_matrix)
        """

        T = self.num_tasks
        total = 0.0

        for i in range(T):
            row_sum = 0.0
            count = 0

            for j in range(i + 1):  # only learned tasks up to i
                val = self.performance_matrix[i, j]

                if np.isnan(val):
                    continue

                row_sum += val
                count += 1

            if count > 0:
                total += (row_sum / count)

        return total / T       

    # ------------------------------------------------------------------
    # UTILITIES
    # ------------------------------------------------------------------

    def print_matrix(self):
        """Pretty-print the full performance matrix."""
        print("\n=== Performance Matrix ===")
        print(f"Rows = learning step, Columns = task being evaluated")
        print(f"(nan = not yet evaluated)\n")

        header = "       " + "  ".join([f"Task{j}" for j in range(self.num_tasks)])
        print(header)
        for i in range(self.num_tasks):
            row_vals = []
            for j in range(self.num_tasks):
                val = self.performance_matrix[i, j]
                row_vals.append(f"{val:.3f}" if not np.isnan(val) else "  nan")
            print(f"Step {i}: " + "  ".join(row_vals))
        print()

    def summary(self):
        """Print a full summary of all recorded steps."""
        print("\n=== Experiment Summary ===")
        for snap in self.history:
            print(f"  Step {snap['step']}: AA={snap['avg_accuracy']:.4f}, "
                  f"FM={snap['forgetting']:.4f}, "
                  f"Scores={[f'{x:.3f}' for x in snap['row']]}")
            print(f"\nAverage Incremental Accuracy:", self.compute_average_incremental_accuracy)


        self.print_matrix()

       
