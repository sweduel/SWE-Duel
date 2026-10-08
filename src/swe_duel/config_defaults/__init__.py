"""Default (template) arena configuration shipped inside the package.

``swe-duel init`` copies these files into a working directory's ``config/`` so the
end user owns an editable configuration (``models.yaml`` menus, gate
thresholds, output paths are all per-deployment choices). The copies here are
read-only templates — nothing at runtime reads them directly.
"""
