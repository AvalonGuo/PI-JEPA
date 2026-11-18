#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""dr3d compose/load: <base scene> + a single distractor object.
Handles default colors and the automatic ground height.

Non-invasive by construction (important): the composed result is "self-contained XML"
(see selfcontain_base_text): <include> expanded in place, meshdir/texturedir made
absolute, so it loads from any path and no longer relies on relative paths that must
sit at the same depth as the real assets; temp files always go to dr3d/tmp/compose/,
third_party is never written (the real assets stay read-only).
"""
import os
import pathlib
import re

import numpy as np

DR3D = pathlib.Path(__file__).resolve().parent
DST = DR3D / 'distractors'
OBJ_XML = {'can': 'can/model.xml', 'bottle': 'bottle/model.xml',
           'cereal': 'cereal/model.xml', 'mug': 'mug/model.xml'}
OBJ_COLOR = {'can': '1 0.5 0 1', 'bottle': '0.2 0.5 0.9 1',
             'cereal': '0.95 0.8 0.2 1', 'mug': '0.85 0.85 0.88 1'}
# extra lift per object (table top z=0), only for objects that look slightly off
OBJ_LIFT = {'mug': 0.01}
HAMMER = ('/root/3DP/third_party/rrl-dependencies/mj_envs/mj_envs/'
          'hand_manipulation_suite/assets/DAPG_hammer.xml')
DOOR = ('/root/3DP/third_party/rrl-dependencies/mj_envs/mj_envs/'
        'hand_manipulation_suite/assets/DAPG_door.xml')
PEN = ('/root/3DP/third_party/rrl-dependencies/mj_envs/mj_envs/'
       'hand_manipulation_suite/assets/DAPG_pen.xml')

# adroit catgen base scenes follow the task; door and hammer share the include
# structure (one include level under DAPG_Adroit.xml / DAPG_assets.xml, verified)
# metaworld catgen base scenes (task -> main XML)
ASSEMBLY = ('/root/3DP/third_party/Metaworld/metaworld/envs/assets_v2/'
            'sawyer_xyz/sawyer_assembly_peg.xml')
# stick-push shares basic_scene.xml (same table) and xyz_base.xml (same corner2
# camera + the wrapper's cam_pos override) with assembly => identical crop/frustum,
# so P0 carries over verbatim. NB this XML's sliding thermos body is named "object",
# which collides with the inner body of all 5 catgen assets (cube/can/bottle/cereal/
# mug) and of sphere -> the injection must rename on conflict (see
# _rename_conflicting_bodies), otherwise mj_loadXML fails with
# "repeated name in body array".
STICK_PUSH = ('/root/3DP/third_party/Metaworld/metaworld/envs/assets_v2/'
              'sawyer_xyz/sawyer_stick_obj.xml')

# assembly: 5 self-closing <include> (all real); xyz_base.xml's meshdir and
# non-self-closing <include file="shared_config.xml"> live inside a header
# comment (verified) so the real content is fully inlined -> selfcontain_text
# only expanding self-closing includes already matches mujoco's parser
BASE_XML = {'hammer': HAMMER, 'door': DOOR, 'pen': PEN, 'assembly': ASSEMBLY,
            'stick_push': STICK_PUSH}
COMPOSE_TMP = DR3D / 'tmp' / 'compose'      # temp dir for composed files (inside dr3d)


def _read_text(path):
    return pathlib.Path(path).read_text(encoding='utf-8', errors='ignore')


def abs_mesh_paths(xml_text, base):
    return re.sub(r'file="([^"]+)"',
                  lambda m: 'file="' + os.path.abspath(
                      os.path.join(str(base), m.group(1))) + '"',
                  xml_text)


def extract_assets_and_worldbody(xml_text):
    def inner(tag):
        m = re.search(rf'<{tag}>([\s\S]*?)</{tag}>', xml_text)
        return m.group(1) if m else ''
    return inner('asset'), inner('worldbody')


def _root_inner(xml_text):
    """Children of the root element (e.g. <mujocoinclude>...</mujocoinclude>), for
    include splicing."""
    m = re.search(r'<([A-Za-z_][\w.:-]*)((?:\s[^<>]*?)?)>([\s\S]*)</\s*\1\s*>',
                  xml_text)
    if m is None:
        raise ValueError('root element not found for include expansion')
    return m.group(3)


def _abs_dir_attr(xml_text, base_dir):
    """meshdir/texturedir relative path -> absolute path (resolved against base_dir)."""
    def _rep(m):
        val = m.group(2)
        if os.path.isabs(val):
            return m.group(0)
        return '%s="%s"' % (m.group(1),
                            os.path.normpath(os.path.join(str(base_dir), val)))
    return re.sub(r'(meshdir|texturedir)\s*=\s*["\']([^"\']+)["\']', _rep, xml_text)


def _abs_asset_file_attr(xml_text, base_dir):
    """Explicit relative file= (mesh/texture/hfield) -> absolute.

    Resolve relative to base_dir (the top-level model dir) -- that is where
    mujoco looks once <include> textually splices content in. Two exceptions:
      * <compiler meshdir="..."/> in effect -> mesh file= resolves against
        meshdir (document-global, last one wins); DAPG_assets.xml sets
        meshdir='../../../dependencies/Adroit/resources/meshes/' so adroit
        bare names (forearm_simple.stl) land under that dir.
      * <compiler texturedir="..."/> in effect -> texture file= likewise.
    metaworld assembly has no real compiler meshdir/texturedir (the only
    <compiler meshdir> lives inside a header comment), so there file= resolves
    against base_dir -- exactly the envs_v2/objects/meshes/... layout.
    <include> is NOT handled here: it is expanded (or left as an ignored
    non-self-closing tag) by selfcontain_text.
    """
    def _compiler_dir(attr):
        hits = re.findall(
            r'<compiler\b[^>]*?\b%s\s*=\s*["\']([^"\']+)["\']' % attr,
            xml_text, re.I)
        if not hits:
            return None
        v = hits[-1]
        if os.path.isabs(v):
            return os.path.normpath(v)
        return os.path.normpath(os.path.join(str(base_dir), v))

    meshdir = _compiler_dir('meshdir')
    texturedir = _compiler_dir('texturedir')

    def _rep(mm):
        tag, kind, attrs, close = (mm.group(1), mm.group(2),
                                   mm.group(3), mm.group(4))
        def _f(mm2):
            val = mm2.group(1)
            if os.path.isabs(val):
                return mm2.group(0)
            if kind == 'mesh' and meshdir is not None:
                root = meshdir
            elif kind == 'texture' and texturedir is not None:
                root = texturedir
            else:
                root = str(base_dir)
            return 'file="%s"' % os.path.normpath(os.path.join(root, val))
        attrs = re.sub(r'file\s*=\s*"([^"]+)"', _f, attrs)
        return tag + attrs + close + '>'
    return re.sub(r'(<(mesh|texture|hfield)\b)([^>]*?)(/?)>', _rep, xml_text)


def selfcontain_text(xml_text, base_dir, max_depth=4):
    """<include> expanded in place (relative to base_dir) + meshdir/texturedir made
    absolute.

    mujoco's <include> is a textual splice: the included file's root element (e.g.
    <mujocoinclude>) has its children replace the include tag itself. DAPG_* scenes
    have only one include level, no nesting; the loop just guards future nesting.
    """


    base_dir = str(base_dir)

    def _expand(m):
        inc = pathlib.Path(base_dir) / m.group(1)
        if not inc.is_file():
            # e.g. include inside an XML comment (metaworld header) or a
            # legitimately optional file: leave the tag untouched -> mujoco
            # ignores it the same way (comments are skipped by the parser).
            return m.group(0)
        # mujoco <include> is a TEXTUAL splice: after expansion every relative
        # file= path resolves against the TOP-LEVEL model dir (base_dir), NOT
        # the included file's dir (verified on metaworld: objects/assets/*.xml
        # use ../objects/meshes/... which must land in envs_v2/objects/meshes).
        inner = _root_inner(_read_text(inc))
        return _abs_dir_attr(_abs_asset_file_attr(inner, base_dir), base_dir)

    for _ in range(max_depth):
        new = re.sub(r'<include\s+file\s*=\s*["\']([^"\']+)["\']\s*/>',
                     _expand, xml_text)
        if new == xml_text:
            break
        xml_text = new
    return _abs_dir_attr(_abs_asset_file_attr(xml_text, base_dir), base_dir)


def selfcontain_base_text(base_xml):
    """Base scene main XML (DAPG_hammer.xml / DAPG_door.xml) -> self-contained text."""
    p = pathlib.Path(base_xml)
    return selfcontain_text(_read_text(p), p.parent)


def object_ground_z(xml_path):
    """Load the object standalone and return the mesh world min z
    (relative to the body origin)."""
    from mujoco_py import load_model_from_path
    m = load_model_from_path(str(xml_path))
    v = np.asarray(m.mesh_vert)
    if v.size and v.size % 3 == 0:
        pts = v.reshape(-1, 3)
        return float(pts[:, 2].min())
    mins = []
    for g in range(m.ngeom):
        sz = m.geom_size[g]
        if len(sz) > 2 and m.geom_type[g] == 6:  # 6 = box
            mins.append(float(m.geom_pos[g, 2]) - float(sz[2]))
        elif m.geom_type[g] == 2:                # 2 = sphere (radius in size[0])
            mins.append(float(m.geom_pos[g, 2]) - float(sz[0]))
    return float(min(mins)) if mins else 0.0


def _floor_body_mass(scene, boundmass="1e-07", boundinertia="1e-11"):
    """Append boundmass/boundinertia (tiny lower bounds) to every `<compiler>`.

    Why: the metaworld base scene has `<compiler inertiafromgeom="auto"
    inertiagrouprange="4 5"/>`, so only geom groups 4/5 contribute body inertia; a
    multi-body asset injected as STATIC (free=False, not mocap) -- e.g. the inspire
    left hand in dr3d/distractors/inspire: 12 hinges + geom group=1 -- then gets 0
    mass / 0 inertia and mj_loadXML fails with
    "mass and inertia of moving bodies must be positive".
    boundmass/boundinertia only raise bodies below the bound; the base scene's
    existing bodies (~1e-4 .. 1e2 kg) stay untouched (single-body geoms need no group
    change). Compilers that already carry the attribute are kept as-is (idempotent).
    """
    def _p(m):
        inner = m.group(1).rstrip()
        if "boundmass" in inner:
            return m.group(0)
        add = (f' boundmass="{boundmass}" boundinertia="{boundinertia}"')
        if inner.endswith("/"):          # self-closing <compiler .../>
            inner = inner[:-1].rstrip() + add + "/"
        else:                            # <compiler ...></compiler>
            inner = inner + add
        return "<compiler" + inner + ">"
    return re.sub(r"<compiler\b([^>]*)>", _p, scene)


_NAME_ATTR = r'<%s\b[^>]*\bname\s*=\s*"([^"]+)"'
_XML_COMMENT = re.compile(r'<!--[\s\S]*?-->')
# Tags of the injected body that the base may reference: when a body is renamed
# (<weld body1=...>, etc.) its references must be fixed up too; for every other tag
# a rename would break <actuator joint=>/<equality>, so a clash raises instead of
# being rewritten silently.
_REF_TAGS = ('geom', 'site', 'joint', 'material', 'mesh', 'texture')


def _tag_names(xml_text, tag):
    """Collect X from <tag ... name="X"> (XML comments stripped, so sample names
    inside comments are not taken for real ones)."""
    return set(re.findall(_NAME_ATTR % tag, _XML_COMMENT.sub('', xml_text)))


def _rename_conflicting_bodies(world_in, scene, obj_name):
    """Rename injected bodies that CLASH with a base body name (references fixed up
    too); a clash on any other tag raises.

    metaworld sawyer_stick_obj.xml's sliding thermos body is called "object" and the
    inner body of all 5 catgen assets + sphere is also "object" -> a plain injection
    triggers mj_loadXML "repeated name in body array ... Object name = object".
    Only clashing entries are touched: hammer/door/pen/assembly have none => their
    composed text stays byte-identical (the existing dr3d/tmp/scenes caches and shas
    stay valid, existing suite dynamics unchanged).
    """
    taken = _tag_names(scene, 'body')
    for old in sorted(_tag_names(world_in, 'body') & taken):
        new = 'dr_%s_%s' % (obj_name, old)
        while new in taken:
            new += '_'
        taken.add(new)
        world_in = re.sub(r'(<body\b[^>]*\bname\s*=\s*")%s(")' % re.escape(old),
                          r'\g<1>' + new + r'\g<2>', world_in)
        world_in = re.sub(r'(\bbody\s*=\s*")%s(")' % re.escape(old),
                          r'\g<1>' + new + r'\g<2>', world_in)
    for tag in _REF_TAGS:
        clash = _tag_names(world_in, tag) & _tag_names(scene, tag)
        if clash:
            raise ValueError(
                'injected %s name collides with base scene (%s): %s'
                % (tag, obj_name, sorted(clash)))
    return world_in


def compose_hammer_object(obj_name, xml_path, pos, base_xml=None, scale=1.0,
                          mass=0.1, diag=1e-4, mocap=False, free=True):
    """Compose the self-contained xml for <base> + one object; returns the path.

    x/y come from pos, z is auto-grounded; meshes are multiplied by scale; every
    object geom gets the default color.
    base_xml: base scene main XML (DAPG_hammer.xml / DAPG_door.xml); defaults to
    HAMMER for backward compatibility.
    Self-contained => loads from any path; the temp file goes to COMPOSE_TMP (inside
    dr3d, third_party untouched)."""
    base = pathlib.Path(base_xml) if base_xml is not None else pathlib.Path(HAMMER)
    obj_txt = _read_text(xml_path)
    obj_dir = pathlib.Path(xml_path).parent
    obj_txt = abs_mesh_paths(obj_txt, obj_dir)
    obj_txt = _abs_dir_attr(obj_txt, obj_dir)   # defensive: distractor with a relative meshdir
    assets_in, world_in = extract_assets_and_worldbody(obj_txt)

    # per-mesh scale (multiplied by --scale), avoiding a global shrink of the base
    def _mesh_scale(mm):
        t = mm.group(0)
        m = re.search(r'scale="([0-9.eE+-]+) ([0-9.eE+-]+) ([0-9.eE+-]+)"', t)
        base_s = [float(x) for x in m.groups()] if m else [1.0, 1.0, 1.0]
        ns = ' '.join('%.6g' % (b * float(scale)) for b in base_s)
        if m:
            t = t[:m.start()] + 'scale="%s"' % ns + t[m.end():]
        else:
            t = t[:-2] + ' scale="%s" />' % ns
        return t
    assets_in = re.sub(r'<mesh\b[^>]*/>', _mesh_scale, assets_in)

    # default color: drop the old rgba/material, then give every geom one flat color
    # fallback: keep geoms that already carry material/rgba, only fill colorless ones
    color = OBJ_COLOR.get(obj_name, '1 0.5 0 1')
    def _color(mm):
        t = mm.group(0)
        if re.search(r'\s+(material|rgba)="[^"]*"', t):
            return t
        return re.sub(r'<geom ', '<geom rgba="%s" ' % color, t, count=1)
    world_in = re.sub(r'<geom\b', _color, world_in)

    # auto ground height: z = -min_z(mesh) + 2mm + per-object extra lift
    ground = object_ground_z(xml_path)
    xy = pos.split()
    z = -ground + 0.002 + float(OBJ_LIFT.get(obj_name, 0.0))
    body_pos = '%s %s %.4f' % (xy[0], xy[1], z)

    scene = selfcontain_base_text(base)
    # An injected body that clashes with a base body name (typically sawyer_stick_obj's
    # thermos body "object" vs the catgen asset inner body "object") makes mj_loadXML
    # fail; only clashing entries are renamed, and without a clash the composed text
    # is byte-identical (hammer/door/pen/assembly caches keep hitting).
    world_in = _rename_conflicting_bodies(world_in, scene, obj_name)
    if '<asset>' in scene:
        scene = scene.replace('</asset>', assets_in + '</asset>', 1)
    else:
        m = re.search(r'<mujoco[^>]*>', scene)
        cut = m.end()
        scene = scene[:cut] + '<asset>' + assets_in + '</asset>' + scene[cut:]
    # Joint injection mode (all keep nbody/nmocap predictable):
    #   free=True (default): a <freejoint> -> dynamic, pushable distractor.
    #   mocap=True        : kinematic body (no joint; driven by data.mocap_pos),
    #                       used by the dynamic-distractor suite (dynsphere).
    #   free=False        : STATIC welded body (no joint, no mocap). Required for
    #                       base envs that assume a single mocap body and/or a
    #                       fixed object qpos index -- metaworld sawyer_xyz_env
    #                       hardcodes the object at qpos[9:12] (a <freejoint>
    #                       shifts it and corrupts the arm reset) and assumes a
    #                       single mocap (a 2nd mocap body breaks set_xyz_action).
    mocap_attr = ' mocap="true"' if mocap else ''
    if mocap or not free:
        joint = ''
    else:
        joint = f'<freejoint name="dr_{obj_name}_free"/>'
    body = (f'<body name="dr_{obj_name}"{mocap_attr} pos="{body_pos}" quat="1 0 0 0">'
            f'{joint}'
            f'<inertial pos="0 0 0.02" mass="{mass:g}" '
            f'diaginertia="{diag:g} {diag:g} {diag:g}"/>' + world_in + '</body>')
    # Append to the LAST </worldbody> (not the first!): metaworld assembly expands to
    # 3 <worldbody> blocks (from basic_scene and other includes). Appending to the
    # first would give the injected joints qpos[0:...] and push sawyer back, which
    # collides with the qpos[9:12] object index hardcoded in sawyer_xyz_env and
    # corrupts the arm reset. Appending to the last keeps the injected joints at the
    # end of qpos. adroit's hammer/door expand to a single worldbody, so their output
    # stays byte-identical under this change.
    head, sep, tail = scene.rpartition('</worldbody>')
    if sep:
        scene = head + body + sep + tail
    else:
        scene = scene.replace('</worldbody>', body + '</worldbody>', 1)
    if not free and not mocap:
        # STATIC (non-mocap) injection: the base scene's inertiagrouprange may exclude
        # the injected geom group -> 0 mass/inertia error; see _floor_body_mass.
        # WARNING: patch only when the injected asset brings its own joints (which create
        #   moving bodies; typically dr3d dynhand's inspire left hand). Joint-free
        #   distractors (catgen/posgen/dynsphere cube/sphere/...) already load fine, and
        #   adding boundmass/boundinertia would raise the base scene's mass~0 bodies
        #   (e.g. the manipulated RoundNut, right_hand) to 1e-7 -- needlessly changing
        #   existing suite dynamics. So skip it, keeping the composed text byte-identical.
        #   adroit's mocap/free paths never reach this branch, behaviour unchanged.
        if "<joint" in world_in:
            scene = _floor_body_mass(scene)
    COMPOSE_TMP.mkdir(parents=True, exist_ok=True)
    tmp = COMPOSE_TMP / ('dr3d_combo_%s_%d.xml' % (obj_name, os.getpid()))
    tmp.write_text(scene, encoding='utf-8')
    return str(tmp)


def _info(m, tag):
    print(f'[OK] {tag}: nbody={m.nbody} ngeom={m.ngeom} nmesh={m.nmesh} '
          f'nfree={m.nq - m.nv}')


def main():
    import mujoco_py
    from mujoco_py import load_model_from_path
    print('== standalone load ==')
    for name, p in OBJ_XML.items():
        m = load_model_from_path(str(DST / p))
        _info(m, f'standalone {name}')
    print('== compose base + object ==')
    for base_name, base_path in BASE_XML.items():
        for name, p in OBJ_XML.items():
            tmp = compose_hammer_object(name, DST / p, '-0.22 -0.15 0.03',
                                        base_xml=base_path)
            try:
                m = load_model_from_path(tmp)
                _info(m, f'{base_name} + {name}')
            finally:
                os.remove(tmp)
    print('done')


if __name__ == '__main__':
    main()
