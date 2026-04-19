class HarmonicCadenceAdapter:
    """
    Adapter for musical voice-leading and harmonic resolution.
    State is the current chord (e.g., F# diminished 7th).
    Home is the Tonic chord of the current key (e.g., C Major).
    """

    def get_available_moves(self) -> list:
        # Valid musical movements based on classical/jazz voice leading
        return [
            "CIRCLE_OF_FIFTHS_DESCEND", # The strongest musical movement (e.g., G7 -> C)
            "STEPWISE_BASS_UP",         # Move the lowest note up one scale degree
            "STEPWISE_BASS_DOWN",       # Move the lowest note down one scale degree
            "TRITONE_SUBSTITUTION",     # Jazz move: swap dominant for its tritone
            "RELATIVE_MINOR_SHIFT"      # e.g., C Major to A Minor
        ]

    def apply_move(self, current_chord, move):
        # Applies a music theory transformation to the current chord
        # (Assuming a helper function `transform_chord` handles the music math)
        return transform_chord(current_chord, move)

    def is_home(self, current_chord) -> bool:
        # We are "home" when the tension is perfectly resolved to the I chord
        return current_chord.name == self.target_tonic.name and current_chord.inversion == 0

    def get_inverse(self, current_chord) -> list:
        """
        Given we are stuck on a weird chord, what is the sequence of
        harmonic moves to resolve gracefully back to the tonic?
        """
        # If we are on a dominant chord, one move fixes it.
        if is_dominant_of(current_chord, self.target_tonic):
            return ["CIRCLE_OF_FIFTHS_DESCEND"]

        # If we are completely lost in a different key, apply a sequence:
        # Shift to a pivot chord, then use the circle of fifths to get home.
        if is_distant_key(current_chord, self.target_tonic):
            return ["TRITONE_SUBSTITUTION", "CIRCLE_OF_FIFTHS_DESCEND", "CIRCLE_OF_FIFTHS_DESCEND"]

        return ["STEPWISE_BASS_DOWN"] # Default safe movement
