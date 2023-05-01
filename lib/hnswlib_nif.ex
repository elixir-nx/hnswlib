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

  def new(
        _space,
        _dim,
        _max_elements,
        _m,
        _ef_construction,
        _random_seed,
        _allow_replace_deleted
      ),
      do: :erlang.nif_error(:not_loaded)

  def knn_query(_self, _data, _k, _num_threads, _filter, _rows, _features),
    do: :erlang.nif_error(:not_loaded)

  def add_items(_self, _f32_data, _ids, _num_threads, _replace_deleted, _rows, _features),
    do: :erlang.nif_error(:not_loaded)

  def get_ids_list(_self), do: :erlang.nif_error(:not_loaded)

  def get_ef(_self), do: :erlang.nif_error(:not_loaded)

  def set_ef(_self, _new_ef), do: :erlang.nif_error(:not_loaded)

  def mark_deleted(_self, _label), do: :erlang.nif_error(:not_loaded)

  def unmark_deleted(_self, _label), do: :erlang.nif_error(:not_loaded)

  def resize_index(_self, _new_size), do: :erlang.nif_error(:not_loaded)

  def get_max_elements(_self), do: :erlang.nif_error(:not_loaded)

  def get_current_count(_self), do: :erlang.nif_error(:not_loaded)

  def float_size, do: :erlang.nif_error(:not_loaded)
end
