from abc import ABC, abstractmethod, abstractproperty
from mercury.db_api.data import Data2D, Data3D, Data4D, Meta
import pandas as pd
from typing import List, Literal, Optional, Tuple
import numpy as np
import json
from pathlib import Path
import re
from copy import deepcopy
import pint

from mercury.db_api.exceptions import MercuryDBException
from mercury.db_api.io import verify_file_exists, load_run_from_csv, load_button_quant_from_csv
from mercury.db_api.units import units

class AbstractMercuryDBAPI(ABC):
    """Abstract base class defining the Mercury Database API interface."""
    def __init__(self):
        """Initializes the abstract database connection."""
        pass

class LocalMercuryDBAPI(AbstractMercuryDBAPI):
    """In-memory database API for managing and querying Mercury assay datasets.

    Manages multi-dimensional dataset structures (Data4D, Data3D, Data2D) loaded from CSV files
    for kinetics, binding isotherms, standard curves, and button quantification.
    """

    def __init__(
        self,
        standard_curve_data_path: Optional[str] = None,
        standard_name: Optional[str] = None,
        standard_substrate: Optional[str] = None,
        standard_units: Optional[pint.Unit] = None,
        standard_concentration_col: Optional[str] = None,
        kinetic_data_path: Optional[str] = None,
        kinetic_name: Optional[str] = None,
        kinetic_substrate: Optional[str] = None,
        kinetic_units: Optional[pint.Unit] = None,
        kinetic_concentration_col: Optional[str] = None,
        time_units: Optional[pint.Unit] = None,
        button_quant_data_path: Optional[str] = None,
    ):
        """Initializes LocalMercuryDBAPI instance.

        Supports legacy positional keyword initialization or clean zero-argument initialization
        followed by explicit `load_run()` calls.

        Args:
            standard_curve_data_path (str, optional): CSV file path for standard curve data.
            standard_name (str, optional): Run identifier for standard curve data.
            standard_substrate (str, optional): Substrate name for standard curve.
            standard_units (pint.Unit, optional): Concentration units for standard curve.
            standard_concentration_col (str, optional): CSV column name containing concentrations.
            kinetic_data_path (str, optional): CSV file path for kinetic assay data.
            kinetic_name (str, optional): Run identifier for kinetic assay.
            kinetic_substrate (str, optional): Substrate name for kinetic assay.
            kinetic_units (pint.Unit, optional): Concentration units for kinetic assay.
            kinetic_concentration_col (str, optional): CSV column name for kinetic concentrations.
            time_units (pint.Unit, optional): Time units for kinetic assay.
            button_quant_data_path (str, optional): CSV file path for button quantification.
        """
        super().__init__()
        self._init_json_dict()

        legacy_args = [
            standard_curve_data_path, standard_name, standard_units, standard_concentration_col,
            kinetic_data_path, kinetic_name, kinetic_units, kinetic_concentration_col,
            time_units, button_quant_data_path,
        ]
        if any(arg is not None for arg in legacy_args):
            if not all(arg is not None for arg in legacy_args):
                raise MercuryDBException(
                    "Legacy LocalMercuryDBAPI initialization requires all standard, kinetic, "
                    "and button_quant arguments to be provided."
                )
            self.load_run(
                standard_name,
                standard_curve_data_path,
                'kinetics',
                conc_unit=standard_units,
                time_unit=time_units,
                concentration_col=standard_concentration_col,
            )
            self.load_run(
                kinetic_name,
                kinetic_data_path,
                'kinetics',
                conc_unit=kinetic_units,
                time_unit=time_units,
                concentration_col=kinetic_concentration_col,
            )
            self.load_run('button_quant', button_quant_data_path, 'button_quant')

    def load_run(
        self,
        run_name: str,
        csv_path: str,
        run_type: Literal["kinetics", "binding", "button_quant"],
        *,
        conc_unit: pint.Unit = None,
        time_unit: pint.Unit = None,
        concentration_col: str = None,
        signal_col: str = None,
        image_type_col: str = None,
        post_wash_prey_type: str = None,
        post_wash_bait_type: str = None,
    ) -> None:
        """Load an assay dataset from CSV and ingest it into the local database.

        Args:
            run_name (str): Identifier name to assign to the ingested run.
            csv_path (str): File path to input CSV dataset.
            run_type (Literal['kinetics', 'binding', 'button_quant']): Type of assay data contained in CSV.
            conc_unit (pint.Unit, optional): Concentration units (e.g. uM).
            time_unit (pint.Unit, optional): Time units (e.g. s).
            concentration_col (str, optional): CSV column name containing concentration values.
            signal_col (str, optional): CSV column name containing raw signal intensity.
            image_type_col (str, optional): CSV column name specifying image type metadata.
            post_wash_prey_type (str, optional): Image type descriptor for post-wash prey fluorescence.
            post_wash_bait_type (str, optional): Image type descriptor for post-wash bait fluorescence.

        Raises:
            MercuryDBException: If the file does not exist or CSV parsing fails.
        """
        verify_file_exists(csv_path)

        if run_type in ('kinetics', 'binding'):
            if conc_unit is None or time_unit is None or concentration_col is None:
                raise MercuryDBException(
                    f"run_type '{run_type}' requires conc_unit, time_unit, and concentration_col."
                )
            run_data = load_run_from_csv(
                csv_path,
                run_type,
                conc_unit,
                time_unit,
                concentration_col,
                signal_col=signal_col,
                image_type_col=image_type_col,
                post_wash_prey_type=post_wash_prey_type,
                post_wash_bait_type=post_wash_bait_type,
            )
        elif run_type == 'button_quant':
            run_data = load_button_quant_from_csv(csv_path)
        else:
            raise MercuryDBException(f"Unknown run_type '{run_type}'.")

        self.add_run(run_name, run_data)


    def _init_json_dict(self) -> None:
        '''
        Populates an initial dictionary with chamber specific metadata.

        Parameters:
                None

        Returns:
                None
        ''' 
        self._json_dict = dict()
        self._json_dict["metadata"] = dict() # Will contain chamber_IDs, sample_IDs as 1D numpy arrays of shape (n_chambers, )
        self._json_dict["runs"] = dict()


    def __repr__(self) -> str:
        '''
        Returns a string representation of the object.

        Returns:
                str: A string representation of the object
        '''
        def recursive_string(d: dict, indent: int, width=5) -> str:
            s = ""

            # How many keys are in the dictionary?
            num_keys = len(d)

            # If there are more than 20 keys, we're in a data-dense region. Let's only show the first 5.
            if num_keys > 20: 
                truncate=True 
            else: 
                truncate=False

            for i, (key, value) in enumerate(d.items()):
                if i > 5 and truncate:
                    s += "\t" * indent + "...\n"
                    break
                s += "\t" * indent + str(key) + ": "
                if isinstance(value, dict):
                    s += "\n" + recursive_string(value, indent + 1)
                else:
                    data_string = ""
                    data_string += str(type(value)) + " "
                    if isinstance(value, np.ndarray):
                        data_string += str(value.shape) + " "
                    value_string = str(value)
                    value_string = value_string.replace("\n", " ").replace("\t", " ")
                    if len(value_string) > 30:
                        data_string += value_string[:30] + "..."
                    else:
                        data_string += value_string
                    s += f"{data_string}\n"
            s += "\t"*indent + '}\n'
            return s
        
        return recursive_string(self._json_dict, 0)

    ### GETTERS & SETTERS
    def add_run(self, run_name: str, run_data) -> None:
        '''
        Adds a run to the database.

                Parameters:
                        run_name (str): Name of the run
                        run_data (Data4D, Data3D, etc): Data for the run

                Returns:
                        None
        '''
        # Check if the format matches one of the allowed dataclasses:
        if not isinstance(run_data, (Data2D, Data3D, Data4D)):
            raise MercuryDBException(f"Run data must be of type Data2D, Data3D, or Data4D. Got {type(run_data)}")
        
        # Add to the database:
        self._json_dict['runs'][run_name] = run_data
        return
    
    def get_run(self, run_name: str) -> dict:
        '''
        Gets a run from the database.

                Parameters:
                        run_name (str): Name of the run

                Returns:
                        dict: Data for the run
        '''
        if run_name not in self._json_dict['runs'].keys():
            raise MercuryDBException(f"Run {run_name} not found in database.")
        
        return self._json_dict['runs'][run_name]
    
    def get_metadata(self, name: str) -> dict:
        '''
        Gets metadata from the database.

                Parameters:
                        name (str): Name of the metadata

                Returns:
                        dict: Metadata
        '''
        # TODO: unused
        if name not in self._json_dict['metadata'].keys():
            raise MercuryDBException(f"Metadata {name} not found in database.")
        
        return self._json_dict['metadata'][name]
    
    def get_run_names(self):
        '''
        Gets the names of all runs in the database.
            
        Parameters:
            None
        Returns:
            list: List of run names
        '''
        # TODO: This should also return the run names of the runs in db_conn
        return [key for key in self._json_dict['runs'].keys()]

    def set_metadata(self, name: str, value: str) -> None:
        '''
        Sets metadata in the database.

                Parameters:
                        name (str): Name of the metadata
                        value (str): Value of the metadata

                Returns:
                        None
        '''
        # TODO: unused
        self._json_dict['metadata'][name] = value

         
    # def export_json(self):
    #     '''This writes the database to file, as a dict -> json'''
    #     with open('db.json', 'w') as fp:
    #         json.dump(self._json_dict, fp, indent=4)
