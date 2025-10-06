{ pkgs }: {
  deps = [
    pkgs.python310
    pkgs.python310Packages.pip
    pkgs.python310Packages.numpy
    pkgs.python310Packages.pandas
    pkgs.python310Packages.scikit-learn
    pkgs.python310Packages.plotly
    pkgs.python310Packages.reportlab
    pkgs.python310Packages.streamlit
  ];
}
