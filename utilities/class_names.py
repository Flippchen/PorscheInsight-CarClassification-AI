from typing import List

CAR_TYPE = ['718 Boxster', '718 Cayman', '911', '918', 'Boxster', 'Carrera Gt', 'Cayenne', 'Cayman', 'Macan', 'Panamera']

ALL_MODEL_VARIANTS = ['718 Boxster_2016', '718 Boxster_2017', '718 Boxster_2018', '718 Boxster_2019', '718 Cayman_2016', '718 Cayman_2017', '718 Cayman_2018', '718 Cayman_2019', '911_1980', '911_1990', '911_2001', '911_2003', '911_2004', '911_2006', '911_2007', '911_2008', '911_2009', '911_2012', '911_2013', '911_2014', '911_2015', '911_2016', '911_2017', '911_2018', '911_2019', '918_2015', 'Boxster_2000', 'Boxster_2001', 'Boxster_2002', 'Boxster_2003', 'Boxster_2004', 'Boxster_2005', 'Boxster_2006', 'Boxster_2007', 'Boxster_2008', 'Boxster_2009', 'Boxster_2010', 'Boxster_2013', 'Boxster_2014', 'Boxster_2015', 'Boxster_2016', 'Boxster_2019', 'Carrera Gt_2004', 'Cayenne_2003', 'Cayenne_2004', 'Cayenne_2005', 'Cayenne_2006', 'Cayenne_2008', 'Cayenne_2009', 'Cayenne_2010', 'Cayenne_2011', 'Cayenne_2012', 'Cayenne_2013', 'Cayenne_2014', 'Cayenne_2015', 'Cayenne_2016', 'Cayenne_2017', 'Cayenne_2018', 'Cayenne_2019', 'Cayman_2005', 'Cayman_2006', 'Cayman_2007', 'Cayman_2008', 'Cayman_2009', 'Cayman_2010', 'Cayman_2011', 'Cayman_2012', 'Cayman_2013', 'Cayman_2014', 'Cayman_2015', 'Cayman_2016', 'Macan_2014', 'Macan_2015', 'Macan_2016', 'Macan_2017', 'Macan_2018', 'Macan_2019', 'Panamera_2009', 'Panamera_2010', 'Panamera_2011', 'Panamera_2012', 'Panamera_2013', 'Panamera_2014', 'Panamera_2015', 'Panamera_2016', 'Panamera_2017', 'Panamera_2018', 'Panamera_2019']

MODEL_VARIANTS = ['911_930', '911_964', '911_991', '911_991_Facelift', '911_992', '911_996_Facelift', '911_997', '911_997_Facelift', '918_Spyderr_1_Generation', 'Boxster_981', 'Boxster_982', 'Boxster_986', 'Boxster_987_Facelift', 'Boxter_986_Facelift', 'Boxter_987', 'Carrera_GT_980', 'Cayenne_955', 'Cayenne_955_Facelift', 'Cayenne_958', 'Cayenne_958_Facelift', 'Cayenne_9YA', 'Cayman_981C', 'Cayman_982C', 'Cayman_987C', 'Cayman_987_Facelift', 'Macan_95B', 'Macan_95B_Facelift', 'Panamera_970', 'Panamera_970_Facelift', 'Panamera_971']


def get_classes_for_model(name: str) -> List[str]:
    """
    Args:
        name (str): The name of the model for which the class names are required.
                    Accepted values are "car_type", "all_specific_model_variants",
                    and "specific_model_variants".

    Returns:
        List[str]: A list of strings containing class names for a specified model.

    Raises:
        ValueError: If an invalid model name is provided as input.
    """
    if name == "car_type":
        return CAR_TYPE
    elif name == "all_specific_model_variants":
        return ALL_MODEL_VARIANTS
    elif name == "specific_model_variants":
        return MODEL_VARIANTS
    else:
        raise ValueError("Invalid model name")
