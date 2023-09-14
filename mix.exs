defmodule HNSWLib.MixProject do
  use Mix.Project

  @version "0.1.1"
  @github_url "https://github.com/elixir-nx/hnswlib"

  def project do
    [
      app: :hnswlib,
      version: @version,
      elixir: "~> 1.12",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      package: package(),
      docs: docs(),
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_precompiler: {:nif, CCPrecompiler},
      make_precompiler_url: "#{@github_url}/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "hnswlib_nif",
      make_precompiler_nif_versions: [versions: ["2.16", "2.17"]]
    ]
  end

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      # compilation
      {:cc_precompiler, "~> 0.1.0"},
      {:elixir_make, "~> 0.7.0"},

      # runtime
      {:nx, "~> 0.5"},

      # docs
      {:ex_doc, "~> 0.29", only: :docs, runtime: false}
    ]
  end

  defp docs do
    [
      main: "HNSWLib",
      source_ref: "v#{@version}",
      source_url: @github_url
    ]
  end

  defp package() do
    [
      files: ~w(3rd_party/hnswlib c_src lib mix.exs README* LICENSE* Makefile checksum.exs),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @github_url}
    ]
  end
end
