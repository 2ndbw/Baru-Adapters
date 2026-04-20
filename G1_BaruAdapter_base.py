class BaruAdapter:
    def get_available_moves(self) -> list:
        """
        1. What moves exist.
        Returns a finite list of discrete, named moves.
        """
        raise NotImplementedError

    def apply_move(self, current_state, move):
        """
        2. What a move does to your position.
        Takes a state and a move, returns the new state.
        """
        raise NotImplementedError

    def is_home(self, current_state) -> bool:
        """
        3. What 'home' means.
        Evaluates if the current state perfectly matches the target/zero state.
        """
        raise NotImplementedError

    def get_inverse(self, current_state) -> list:
        """
        4. How to get home from any position.
        The hardest part: Given the current state, return the exact sequence
        of moves required to close the gap.
        """
        raise NotImplementedError
