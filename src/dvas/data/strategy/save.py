"""
Copyright (c) 2020 MeteoSwiss, contributors listed in AUTHORS.

Distributed under the terms of the GNU General Public License v3.0 or later.

SPDX-License-Identifier: GPL-3.0-or-later

Module contents: Strategy used to save data

"""

# Import from external packages
from abc import ABCMeta#, abstractmethod

# Import from current package
from ..linker import LocalDBLinker


#class SaveDataStrategy(metaclass=ABCMeta):
#    """Abstract class to manage data saving strategy"""
#
#    def __init__(self):
#        self._local_db_linker = LocalDBLinker()
#
#    @abstractmethod
#    def save(self, *args, **kwargs):
#        """Strategy required method"""
#

class SaveDataStrategy(metaclass=ABCMeta):
    """Class to manage saving of time data"""

    def __init__(self):
        self._local_db_linker = LocalDBLinker()

    def save(self, values, events, df_to_db_keys, add_tags, rm_tags):
        """ Implementation of save method.

        Args:
            values (dict of list of Profiles): the data to save into the database.
            event (dict): event information
            df_to_db_keys (dict of str): names of database parameters associated to the DataFrame
                column names.
            add_tags (list of str): list of tags to add to the event before ingestion by the
                database.
            rm_tags (list of str): list of tags to remove before ingestion by teh database.

        """

        # Loop through the different types of profiles
        for key, data_list in values.items():

            event_list = events[key]

            # Adjust the tags as required
            # TODO: since is not super smart ... maybe a rethink of the tag handling would help.
            for evt_ind, _ in enumerate(event_list):

                if add_tags is not None:
                    for tag in add_tags:
                        event_list[evt_ind].tag_abbr.add(tag)

                if rm_tags is not None:
                    for tag in rm_tags:
                        event_list[evt_ind].tag_abbr.remove(tag)

            # If I get any time delta in the data, change them to floats
            # TODO: I suppose these 3 lines should warrant a new child class for this strategy ?
            for i, _ in enumerate(data_list):
                if 'tdt' in data_list[i].columns:
                    data_list[i] = data_list[i].assign(tdt=data_list[i]['tdt'].dt.total_seconds())

            # TODO: For some reason the time deltas are not stored properly inside the DB.
            # Why is that ? Is the bug here, or elsewhere ?

            # Save to db
            self._local_db_linker.save(
                [
                    {'data': arg_data[col],
                     'event': arg_event,
                     'prm_abbr': df_to_db_keys[key][col],
                    } for arg_data, arg_event in zip(data_list, event_list)
                      for col in arg_data.columns
                ]
            )
