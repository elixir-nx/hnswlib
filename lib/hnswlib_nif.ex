defmodule HNSWLib.Nif do
  @moduledoc false

  @on_load :load_nif
  def load_nif do
    nif_file = ~c"#{:code.priv_dir(:hnswlib_elixir)}/hnswlib_nif"

    case :erlang.load_nif(nif_file, 0) do
      :ok -> :ok
      {:error, {:reload, _}} -> :ok
      {:error, reason} -> IO.puts("Failed to load nif: #{reason}")
    end
  end

  def new(_space, _dim), do: :erlang.nif_error(:not_loaded)

  def init_index(
        _self,
        _max_elements,
        _m,
        _ef_construction,
        _random_seed,
        _allow_replace_deleted
      ),
      do: :erlang.nif_error(:not_loaded)
end
