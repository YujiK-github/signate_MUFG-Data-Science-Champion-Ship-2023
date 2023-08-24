import polars as pl


class CustomOrdinalEncoder(object):
    """
    https://github.com/momijiame/shirokumas/blob/main/shirokumas/_ordinal.py
    https://contrib.scikit-learn.org/category_encoders/ordinal.html
    https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.OrdinalEncoder.html
    """
    def __init__(self, unknown_value:int = 99999, encoded_missing_value:int = -1,):
        self.mappings = {}
        self.unknown_value = unknown_value
        self.encoded_missing_value = encoded_missing_value
        
    def fit(self, df: pl.DataFrame | pl.Series) -> None:
        self.mappings = {}
        if type(df) == pl.DataFrame:
            for col in df.columns:
                unique_values = df[col].unique().to_list()
                self.mappings[col] = {
                    value: i for i, value in enumerate(unique_values)
                }
        else:
            col = df.name
            unique_values = df.unique().to_list()
            self.mappings[col] = {
                value: i for i, value in enumerate(unique_values)
            }
            
    def transform(self, df: pl.DataFrame | pl.Series) -> pl.DataFrame | pl.Series:
        tmp = []
        if type(df) == pl.DataFrame:
            for col in df.columns:
                if None in self.mappings[col].keys():
                    self.mappings[col][None] = self.encoded_missing_value
                tmp.append(pl.col(col).map_dict(self.mappings[col], default=self.unknown_value, return_dtype=pl.Int32).suffix("_category"))
            df = df.with_columns(tmp)
            return df[[col for col in df.columns if "_category" in col]]
        else:
            col = df.name
            if None in self.mappings[col].keys():
                self.mappings[col][None] = self.encoded_missing_value
            tmp.append(pl.col(col).map_dict(self.mappings[col], default=self.unknown_value, return_dtype=pl.Int32).suffix("_category"))
            df = pl.DataFrame(df)
            df = df.with_columns(tmp)      
            return df[[col for col in df.columns if "_category" in col]]  
    
    def fit_transform(self, df: pl.DataFrame | pl.Series) -> pl.DataFrame | pl.Series:        
        self.fit(df)
        return self.transform(df)