#Written by Jacob Monroe, NIST Employee

"""
Set of functions for performing coordinate transformations on molecules.

Many similar functions can be found in other scripts within the repository, or in notebooks.
Fully replacing those functions with these is ongoing work. Mainly, this code provides
access to transformations that are done with tensorflow. This is necessary to make the
computation of potential energies differentiable, since OpenMM requires Cartesian
coordinates as inputs and the VAEs may work with internal coordinates.
"""

import numpy as np
import tensorflow as tf

from MDAnalysis.analysis import bat


#For transforming coordinates for molecular systems...
#Works for any molecular system with dihedrals or sine-cosine pairs last (default)
#Just specifiy the total number of DOFs for the molecule
def sincos(x, totDOFs):
  torsion_sin = np.sin(x[:, -(totDOFs//3 - 3):])
  torsion_cos = np.cos(x[:, -(totDOFs//3 - 3):])
  out_x = np.concatenate([x[:, :-(totDOFs//3 - 3)], torsion_sin, torsion_cos], axis=1)
  return out_x


#Unlike sincos, unsincos just does one config at a time to work with bat_analysis
def unsincos(x, totDOFs):
  if len(x.shape) == 1:
    sin_vals = x[-2*(totDOFs//3 - 3):-(totDOFs//3 - 3)]
    cos_vals = x[-(totDOFs//3 - 3):]
    r_vals = np.sqrt(sin_vals**2 + cos_vals**2)
    torsion_vals = np.arctan2(sin_vals/r_vals, cos_vals/r_vals)
    out_x = np.concatenate([x[:-2*(totDOFs//3 - 3)], torsion_vals])
  else:
    sin_vals = x[:, -2*(totDOFs//3 - 3):-(totDOFs//3 - 3)]
    cos_vals = x[:, -(totDOFs//3 - 3):]
    r_vals = np.sqrt(sin_vals**2 + cos_vals**2)
    torsion_vals = np.arctan2(sin_vals/r_vals, cos_vals/r_vals)
    out_x = np.concatenate([x[:, :-2*(totDOFs//3 - 3)], torsion_vals], axis=1)
  return out_x


#Transformation for united-atom polymer from VAE coordinates to full Cartesian
class transform_poly_tf(object):

    """
    Class to perform transformation from VAE-based coordinates to Cartesian for polymer.
    All implemented in tensorflow.
    Not compatible if working with sine-cosine pairs as inputs/outpus of VAE.
    Only works for periodic degrees of freedom drawn directly from von Mises distributions
    via tensorflow-probability. This makes more sense and works better anyway.
    """

    def __init__(self, bat_to_cart_func=None):

        self.bat_to_cart_func = bat_to_cart_func

        self.totDOFs = 60
        self.bond_inds = list(range(8))
        for b in range(9, 9 + self.totDOFs//3 - 3):
            self.bond_inds.append(b)
        self.bond_mask = np.ones(self.totDOFs, dtype=bool)
        self.bond_mask[self.bond_inds] = False
        self.non_bond_inds = np.arange(self.totDOFs)[self.bond_mask].tolist()

        #Now need to populate values for bonded DOFs
        #Setting rigid translation and rotation (first 6) to zero
        self.bond_vals = tf.concat([tf.zeros(6, dtype=tf.float32),
                                    1.54*tf.ones(len(self.bond_inds) - 6, dtype=tf.float32)],
                                   axis=0)

    def __call__(self, x):
        #To make work with arbitrary batch size, need to tile self.bond_vals
        bond_vals = tf.tile(tf.reshape(self.bond_vals, (1, -1)), (tf.shape(x)[0], 1))
        #And will need to transpose and transpose back so indices work along indended axis
        bat = tf.dynamic_stitch([self.non_bond_inds, self.bond_inds],
                                [tf.transpose(x), tf.transpose(bond_vals)])
        bat = tf.transpose(bat)
        if self.bat_to_cart_func is not None:
            out = self.bat_to_cart_func(bat)
        else:
            out = bat
        return out


#And same for dialanine
class transform_ala_tf(object):

    """
    Class to perform transformation from VAE-based coordinates to Cartesian for dialanine.
    All implemented in tensorflow.
    Not compatible if working with sine-cosine pairs as inputs/outpus of VAE.
    Only works for periodic degrees of freedom drawn directly from von Mises distributions
    via tensorflow-probability. This makes more sense and works better anyway.
    """

    def __init__(self, bat_to_cart_func=None):

        self.bat_to_cart_func = bat_to_cart_func

        self.totDOFs = 66
        self.bond_inds = list(range(6))
        self.bond_inds = self.bond_inds + [9, 11, 12, 13, 15, 17, 18, 19, 21, 24, 25, 26]
        self.bond_mask = np.ones(self.totDOFs, dtype=bool)
        self.bond_mask[self.bond_inds] = False
        self.non_bond_inds = np.arange(self.totDOFs)[self.bond_mask].tolist()

        #Now need to populate values for bonded DOFs
        #Setting rigid translation and rotation (first 6) to zero
        self.bond_vals = tf.constant([0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                                      1.01, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.09, 1.01, 1.09, 1.09, 1.09],
                                     dtype=tf.float32)

    def __call__(self, x):
        #To make work with arbitrary batch size, need to tile self.bond_vals
        bond_vals = tf.tile(tf.reshape(self.bond_vals, (1, -1)), (tf.shape(x)[0], 1))
        #And will need to transpose and transpose back so indices work along indended axis
        bat = tf.dynamic_stitch([self.non_bond_inds, self.bond_inds],
                                [tf.transpose(x), tf.transpose(bond_vals)])
        bat = tf.transpose(bat)
        if self.bat_to_cart_func is not None:
            out = self.bat_to_cart_func(bat)
        else:
            out = bat
        return out


class BAT_tf(bat.BAT):

    """
    Inherits from MDAnalysis bat.BAT,

    https://github.com/MDAnalysis/mdanalysis/blob/develop/package/MDAnalysis/analysis/bat.py

    but adds a function Cartesian_tf that exactly replicates the Cartesian function already
    there, but it's implemented in tensorflow so that gradients can be computed.
    """

    def __init__(self, *args, **kwargs):
        super(BAT_tf, self).__init__(*args, **kwargs)

    def Cartesian_tf(self, bat_frame):
        #Want to be able to operate on multiple frames simultaneously
        #(like a batch), so add dimension if just one configuration
        if len(bat_frame.shape) == 1:
            bat_frame = tf.reshape(bat_frame, (1, -1))
        n_batch = tf.shape(bat_frame)[0]

        # Split the bat vector into more convenient variables
        origin = bat_frame[:, :3]
        (phi, theta, omega) = tf.split(bat_frame[:, 3:6], 3, axis=1)
        (r01, r12, a012) = tf.split(bat_frame[:, 6:9], 3, axis=1)
        n_torsions = (self._ag.n_atoms - 3)
        bonds = bat_frame[:, 9:n_torsions + 9]
        angles = bat_frame[:, n_torsions + 9:2 * n_torsions + 9]
        torsions = bat_frame[:, 2 * n_torsions + 9:]
        # When appropriate, convert improper to proper torsions
        shift = tf.gather(torsions,
                          tf.tile([self._primary_torsion_indices], (n_batch, 1)),
                          batch_dims=1)
        unique_primary_torsion_bool = np.zeros(len(self._primary_torsion_indices), dtype=bool)
        unique_primary_torsion_bool[self._unique_primary_torsion_indices] = True
        shift = tf.where(unique_primary_torsion_bool, x=tf.zeros_like(shift), y=shift)
        torsions += shift
        # Wrap torsions to between -np.pi and np.pi
        torsions = ((torsions + np.pi) % (2 * np.pi)) - np.pi

        # Set initial root atom positions based on internal coordinates
        p0 = tf.zeros((n_batch, 3))
        p1 = tf.transpose(tf.scatter_nd([[2]], [tf.reshape(r01, (-1))], (3, n_batch)))
        p2 = tf.concat([r12 * tf.math.sin(a012),
                        tf.zeros((n_batch, 1)),
                        r01 - r12 * tf.math.cos(a012)], axis=1)

        # Rotate the third atom by the appropriate value
        co = tf.squeeze(tf.math.cos(omega), axis=-1)
        so = tf.squeeze(tf.math.sin(omega), axis=-1)
        # $R_Z(\omega)$
        Romega = tf.transpose(tf.scatter_nd([[0, 0], [0, 1], [1, 0], [1, 1], [2, 2]],
                                            [co, -so, so, co, tf.ones(n_batch)],
                                            (3, 3, n_batch)),
                              perm=(2, 0, 1))
        p2 = tf.squeeze(tf.linalg.matmul(Romega, tf.expand_dims(p2, axis=-1)))
        # Rotate the second two atoms to point in the right direction
        cp = tf.squeeze(tf.math.cos(phi), axis=-1)
        sp = tf.squeeze(tf.math.sin(phi), axis=-1)
        ct = tf.squeeze(tf.math.cos(theta), axis=-1)
        st = tf.squeeze(tf.math.sin(theta), axis=-1)
        # $R_Z(\phi) R_Y(\theta)$
        Re = tf.transpose(tf.scatter_nd([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2], [2, 0], [2, 2]],
                                        [cp*ct, -sp, cp*st, ct*sp, cp, sp*st, -st, ct],
                                        (3, 3, n_batch)),
                          perm=(2, 0, 1))
        p1 = tf.squeeze(tf.linalg.matmul(Re, tf.expand_dims(p1, axis=-1)), axis=-1)
        p2 = tf.squeeze(tf.linalg.matmul(Re, tf.expand_dims(p2, axis=-1)), axis=-1)
        # Translate the first three atoms by the origin
        p0 += origin
        p1 += origin
        p2 += origin

        #With tf, can't change part of a tensor alone, so create list and put together at end
        XYZ = [p0, p1, p2]
        XYZ_order = [self._root_XYZ_inds[0], self._root_XYZ_inds[1], self._root_XYZ_inds[2]]

        # Place the remaining atoms
        for i in range(len(self._torsion_XYZ_inds)):
            (a0, a1, a2, a3) = self._torsion_XYZ_inds[i]
            r01 = bonds[:, i:i+1]
            angle = angles[:, i:i+1]
            torsion = torsions[:, i:i+1]

            p1 = XYZ[XYZ_order.index(a1)]
            p3 = XYZ[XYZ_order.index(a3)]
            p2 = XYZ[XYZ_order.index(a2)]

            sn_ang = tf.math.sin(angle)
            cs_ang = tf.math.cos(angle)
            sn_tor = tf.math.sin(torsion)
            cs_tor = tf.math.cos(torsion)

            v21 = p1 - p2
            v21 /= tf.math.sqrt(tf.reduce_sum(v21 * v21))
            v32 = p2 - p3
            v32 /= tf.math.sqrt(tf.reduce_sum(v32 * v32))

            vp = tf.linalg.cross(v32, v21)
            cs = tf.reduce_sum(v21 * v32)

            sn = tf.math.maximum(tf.math.sqrt(1.0 - cs * cs), 0.0000000001)
            vp = vp / sn
            vu = tf.linalg.cross(vp, v21)

            XYZ.append(p1 + r01*(vu*sn_ang*cs_tor + vp*sn_ang*sn_tor - v21*cs_ang))
            XYZ_order.append(a0)
        XYZ = tf.gather(XYZ, XYZ_order)
        XYZ = tf.transpose(XYZ, perm=(1, 0, 2))
        return XYZ



