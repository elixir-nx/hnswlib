defmodule HNSWLib.MixProject do
  use Mix.Project

  @version "0.1.5"
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
      main: "HNSWLib",
      description: "Elixir binding for the hnswlib library",
      compilers: [:elixir_make] ++ Mix.compilers(),
      make_precompiler: {:nif, CCPrecompiler},
      make_precompiler_url: "#{@github_url}/releases/download/v#{@version}/@{artefact_filename}",
      make_precompiler_filename: "hnswlib_nif",
      make_precompiler_nif_versions: [versions: ["2.16", "2.17"]],
      cc_precompiler: cc_precompiler()
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
      source_ref: "v#{@version}",
      source_url: @github_url
    ]
  end

  defp cc_precompiler do
    extra_options =
      if System.get_env("HNSWLIB_CI_PRECOMPILE") == "true" do
        [
          only_listed_targets: true,
          exclude_current_target: true
        ]
      else
        []
      end

    [cleanup: "cleanup"] ++ extra_options
  end

  defp package() do
    [
      files:
        ~w(3rd_party/hnswlib c_src lib mix.exs README* LICENSE* CMakeLists.txt Makefile checksum.exs),
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => @github_url}
    ]
  end
end
