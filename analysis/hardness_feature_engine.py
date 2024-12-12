import re
import pandas as pd

class HardnessFeatureEngine:
    @staticmethod
    def extract_digit(value):
        """
        Extracts the first digit from a given value using regex.
        
        Parameters:
        -----------
        value : any
            The input value from which to extract a digit.

        Returns:
        --------
        int or None
            The extracted digit, or None if no digit is found.
        """
        match = re.search(r"\d", str(value))  # Search for the first digit
        return int(match.group()) if match else None

    def calculate_combined_hardness_score(self, dataframe, score_columns):
        """
        Calculates the combined hardness score by extracting digits from specific columns
        and averaging them.

        Parameters:
        -----------
        dataframe : pandas.DataFrame
            The input DataFrame containing the score columns.

        score_columns : list
            List of column names from which to extract scores and calculate the combined score.

        Returns:
        --------
        pandas.DataFrame
            The updated DataFrame with extracted digits and the combined hardness score.
        """
        # Extract digits from the specified columns
        for col in score_columns:
            dataframe[col] = dataframe[col].apply(self.extract_digit)

        # Calculate the combined hardness score
        dataframe["combined_hardness_score"] = (
            dataframe[score_columns]
            .mean(axis=1, skipna=True)  # Calculate mean while ignoring NaN values
        )

        return dataframe


