# -*- coding: utf-8 -*-
"""
Created on Thu Dec 13 13:57:58 2018

@author: Charles
"""

import numpy as np

class Indices(dict):
    """This class contains useful functions to compute band ratios
    with image data from the ASTER instrument.

    VNIR:
        Band 1 = index 0
        Band 2 = index 1
        Band 3N = index 2

    SWIR:
        Band 4 = index 3
        Band 5 = index 4
        Band 6 = index 5
        Band 7 = index 6
        Band 8 = index 7
        Band 9 = index 8

    The SpectralIndices object inherits from the dictionary class.
    It is a dictionary-like object whose .get() method was modified to return
    call a function and call itself.

    Arguments
    ==========
    img:    3D numpy array, the stacked bands of the multispectral image
            The shape of stacked images should follow the GDAL convention:
            ->  img.shape = (n_bands, height, width)

    Usage
    =========
    1. Instantiate the SpectralIndices object
        band_ratios = SpectralIndices(img)
    2. Print keys to the available indices
        print(band_ratios.keys())
    3. Call .get() with the desired key
        kaolinite = band_ratios.get("Kaolinite")
    4. The output of .get() is a 3D numpy array
        kaolinite.shape = (1, height, width)
    """
    def __init__(self, img, mask=None):
        # Initialize
        self.img = img
        self.mask = mask

        # Define a mask if None is provided
        # The mask prevents division by 0
        if self.mask is None:
            self.mask = self.img == 0

        self.masked_img = np.ma.masked_array(data=self.img, mask=self.mask)

        # Define keys to VNIR spectral indices
        vnir_indices = {'VI'                :   self._vi,
                        'NDVI'              :   self._ndvi,
                        'STVI'              :   self._stvi,
                        'Ferric_iron'       :   self._ferric_iron,
                        'Ferrous_iron_1'    :   self._ferrous_iron_1,
                        }

        # Define keys to SWIR spectral indices
        swir_indices = {'AlOH'              :   self._al_oh_group,
                        'Laterite'          :   self._laterite,
                        'Alunite'           :   self._alunite,
                        'CCE'               :   self._cce,
                        'Clay_1'            :   self._clay_1,
                        'Clay_2'            :   self._clay_2,
                        'Kaolinitic'        :   self._kaolinitic,
                        'Kaolin_group'      :   self._kaolin_group,
                        'Kaolinite'         :   self._kaolinite,
                        'Muscovite'         :   self._muscovite,
                        'OH_1'              :   self._oh_1,
                        'OH_2'              :   self._oh_2,
                        'OH_3'              :   self._oh_3,
                        'PHI'               :   self._phi,
                        'AKP'               :   self._akp,
                        'Amphibole'         :   self._amphibole,
                        'Calcite'           :   self._calcite,
                        'Dolomite'          :   self._dolomite,
                        'MgOH_group'        :   self._mgoh_group,
                        'MgOH_1'            :   self._mgoh_1,
                        'MgOH_2'            :   self._mgoh_2,
                        }

        # Define keys to other indices
        misc_indices = {'RDB6'              :   self._rdb6,
                        'RDB8'              :   self._rdb8,
                        'Ferrous_iron_2'    :   self._ferrous_iron_2,
                        'Ferric_oxide'      :   self._ferric_oxide,
                        'Gossan'            :   self._gossan,
                        'Opaque_index'      :   self._opaque_index,
                        'Silicates'         :   self._silicates,
                        'Burn_index'        :   self._burn_index,
                        'Salinity'          :   self._salinity,
                        }

        # Merge the various indices groups into self
        self.update(vnir_indices)
        self.update(swir_indices)
        self.update(misc_indices)

    def _reshape_output(self, masked_array):
        """ Utility function to reshape the output spectral indice image to
        shape = (1, height, width).
        This is useful to later write to a GeoTiff with GDA and makes output
        dimensions (ndim = 3) consistent with input dimensions (ndim = 3).
        """
        return masked_array.data[np.newaxis,:]

    def get(self, key):
        """ Override of the normal dictionary .get() method
        We want .get() to return the reshaped (1, height, width) output of a
        call of the desired spectral index function
        """
        return self._reshape_output(dict.__getitem__(self, key)())


    """ VNIR spectral indices
    ----------------------------------------------------------------------------
    """
    def _vi(self):
        """Vegetation Index
        (Pour & Hashim, 2011)
        """
        X = self.masked_img
        return X[2] /  X[1]

    def _ndvi(self):
        """Vegetation Index
        (Rouse et al., 1974)
        """
        X = self.masked_img
        return (X[2] -  X[1]) / (X[2] +  X[1])

    def _stvi(self):
        """Vegetation Index
        (Pour & Hashim, 2011)
        """
        X = self.masked_img
        return (X[2] /  X[1]) * (X[0] /  X[1])

    def _ferric_iron(self):
        """Ferric Iron 3
        (Rowan & Mars, 2003)
        """
        X = self.masked_img
        return X[1] /  X[0]

    def _ferrous_iron_1(self):
        """Ferrous Iron 1
        (Rowan et al., 2005)
        """
        X = self.masked_img
        return X[0] /  X[1]


    """ SWIR spectral indices
    ----------------------------------------------------------------------------
    """
    def _al_oh_group(self):
        """AlOH Group
        (Cudahy, 2012)
        """
        X = self.masked_img
        return X[4] / X[6]

    def _laterite(self):
        """Alteration/Laterite
        (Bierwith, 2002)
        """
        X = self.masked_img
        return X[3] / X[4]

    def _alunite(self):
        """Alteration/Laterite
        (Bierwith, 2002)
        """
        X = self.masked_img
        return (X[6] / X[4]) * (X[6] / X[7])

    def _cce(self):
        """CCE
        (Rowan & Mars, 2003)
        """
        X = self.masked_img
        return (X[7] + X[8]) / X[7]

    def _clay_1(self):
        """Clay 1
        (Rowan & Mars, 2003)
        """
        X = self.masked_img
        return (X[4] + X[6]) / X[5]

    def _clay_2(self):
        """Clay 2
        (Bierwith, 2002)
        """
        X = self.masked_img
        return (X[4] * X[6]) / X[5]**2

    def _kaolinitic(self):
        """Kaolinitic
        (Hewson et al., 2005)
        """
        X = self.masked_img
        return X[6] / X[4]

    def _kaolin_group(self):
        """Kaolin Group
        (Cudahy, 2012)
        """
        X = self.masked_img
        return X[5] / X[4]

    def _kaolinite(self):
        """Kaolin Group
        (Pour & Hashim, 2011)
        """
        X = self.masked_img
        return (X[3] / X[4]) * (X[7] / X[5])

    def _muscovite(self):
        """Muscovite
        (Hewson et al., 2005)
        """
        X = self.masked_img
        return X[6] / X[5]

    def _oh_1(self):
        """OH 1
        (Pour & Hashim, 2011)
        """
        X = self.masked_img
        return (X[6] / X[5]) * (X[3] / X[5])

    def _oh_2(self):
        """OH 2
        (Ninomiya et al., 2005)
        """
        X = self.masked_img
        return (X[3] * X[6] / x[5]) / X[5]

    def _oh_3(self):
        """OH 3
        (Ninomiya et al., 2005)
        """
        X = self.masked_img
        return (X[3] * X[6] / x[4]) / X[4]

    def _phi(self):
        """PHI
        (Hewson et al., 2005)
        """
        X = self.masked_img
        return X[4] / X[5]

    def _akp(self):
        """AKP
        (Rowan & Mars, 2003)
        """
        X = self.masked_img
        return (X[3] + X[5]) / X[4]

    def _amphibole(self):
        """Amphibole
        (Bierwirth, 2002)
        """
        X = self.masked_img
        return X[5] / X[7]

    def _calcite(self):
        """Calcite
        (Pour & Hashim, 2011)
        """
        X = self.masked_img
        return (X[5] / X[7]) * (X[8] / X[7])

    def _dolomite(self):
        """Dolomite
        (Rowan & Mars, 2003)
        """
        X = self.masked_img
        return (X[5] + X[7]) / X[6]

    def _mgoh_group(self):
        """MgOH Group
        (Cudahy, 2012)
        """
        X = self.masked_img
        return (X[5] + X[8]) / (X[6] + X[7])

    def _mgoh_1(self):
        """MgOH 1
        (Hewson et al., 2005)
        """
        X = self.masked_img
        return (X[5] + X[8]) / X[7]

    def _mgoh_2(self):
        """MgOH 2
        (Cudahy, 2012)
        """
        X = self.masked_img
        return X[6] / X[7]


    """ Other spectral indices
    ----------------------------------------------------------------------------
    """
    def _rdb6(self):
        """RDB6
        (Rowan et al., 2005)
        """
        X = self.masked_img
        return (X[3] + X[6]) / (X[5] * X[1])

    def _rdb8(self):
        """RDB6
        (Rowan et al., 2005)
        """
        X = self.masked_img
        return (X[6] + X[8]) / (X[7] * X[1])

    def _rdb8(self):
        """RDB6
        (Rowan et al., 2005)
        """
        X = self.masked_img
        return (X[6] + X[8]) / (X[7] * X[1])

    def _ferrous_iron_2(self):
        """Ferrous Iron 2
        (Rowan & Mars, 2003)
        """
        X = self.masked_img
        return (X[4] /  X[2]) + (X[0] /  X[1])

    def _ferric_oxide(self):
        """Ferric oxide
        (Cudahy, 2012)
        """
        X = self.masked_img
        return X[5] /  X[2]

    def _gossan(self):
        """Gossan
        (Volesky et al., 2012)
        """
        X = self.masked_img
        return X[5] /  X[1]

    def _opaque_index(self):
        """Opaque index
        (Cudahy, 2012)
        """
        X = self.masked_img
        return X[0] /  X[3]

    def _silicates(self):
        """Silicates
        (Cudahy, 2012)
        """
        X = self.masked_img
        return X[4] /  X[3]

    def _burn_index(self):
        """Burn index
        (Hudak et al., 2004)
        """
        X = self.masked_img
        return (X[2] -  X[4]) / (X[2] +  X[5])

    def _salinity(self):
        """Salinity index
        (Al-Khaier, 2003)
        """
        X = self.masked_img
        return (X[3] -  X[4]) / (X[3] +  X[4])
