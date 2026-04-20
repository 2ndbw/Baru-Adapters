class CatalogExtractionAdapter:
    """
    Adapter for correcting failed OCR extractions in a grid-based catalog scan.
    State is a dictionary of cell IDs and their current error status.
    Home is a state where all cells are fully resolved and match the database.
    """

    def get_available_moves(self) -> list:
        # These are the actions the system can take to try and fix a broken cell scan
        return [
            "APPLY_HIGH_CONTRAST_FILTER", # Cost: Low
            "RE_CROP_BOUNDING_BOX",       # Cost: Low
            "FUZZY_MATCH_INDEX",          # Cost: Medium (Search the HO Slot Car Index for closest text)
            "FLAG_FOR_MANUAL_REVIEW"      # Cost: Extremely High (The ultimate fallback)
        ]

    def apply_move(self, current_state, move):
        # Applies the correction move to a specific cell in the grid
        new_state = current_state.copy()
        for cell_id, status in new_state.items():
            if status != "RESOLVED":
                if move == "APPLY_HIGH_CONTRAST_FILTER":
                    new_state[cell_id] = "OCR_RETRY_PENDING"
                elif move == "FUZZY_MATCH_INDEX":
                    new_state[cell_id] = "DATABASE_MATCH_PENDING"
                # ... mapping logic for other moves
        return new_state

    def is_home(self, current_state) -> bool:
        # The loop is perfect when every single cell in the 4x4 grid is verified
        return all(status == "RESOLVED" for status in current_state.values())

    def get_inverse(self, current_state) -> list:
        """
        The heavy lifting. If a cell read 'Aura AFX' instead of 'Aurora AFX',
        what is the exact sequence of moves to fix it?
        """
        moves_needed = []
        for cell_id, error_type in current_state.items():
            if error_type == "TEXT_BLURRY":
                moves_needed.extend(["APPLY_HIGH_CONTRAST_FILTER", "RE_CROP_BOUNDING_BOX"])
            elif error_type == "UNKNOWN_MODEL_NUMBER":
                moves_needed.append("FUZZY_MATCH_INDEX")
            elif error_type == "IMAGE_CORRUPTED":
                moves_needed.append("FLAG_FOR_MANUAL_REVIEW")
        return moves_needed
