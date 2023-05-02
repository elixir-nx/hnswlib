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

  def index_new(
        _space,
        _dim,
        _max_elements,
        _m,
        _ef_construction,
        _random_seed,
        _allow_replace_deleted
      ),
      do: :erlang.nif_error(:not_loaded)

  def index_knn_query(_self, _data, _k, _num_threads, _filter, _rows, _features),
    do: :erlang.nif_error(:not_loaded)

  def index_add_items(_self, _f32_data, _ids, _num_threads, _replace_deleted, _rows, _features),
    do: :erlang.nif_error(:not_loaded)

  def index_get_items(_self, _ids, _return), do: :erlang.nif_error(:not_loaded)

  def index_get_ids_list(_self), do: :erlang.nif_error(:not_loaded)

  def index_get_ef(_self), do: :erlang.nif_error(:not_loaded)

  def index_set_ef(_self, _new_ef), do: :erlang.nif_error(:not_loaded)

  def index_get_num_threads(_self), do: :erlang.nif_error(:not_loaded)

  def index_set_num_threads(_self, _new_num_threads), do: :erlang.nif_error(:not_loaded)

  def index_save_index(_self, _path), do: :erlang.nif_error(:not_loaded)

  def index_load_index(_self, _path, _max_elements, _allow_replace_deleted),
    do: :erlang.nif_error(:not_loaded)

  def index_mark_deleted(_self, _label), do: :erlang.nif_error(:not_loaded)

  def index_unmark_deleted(_self, _label), do: :erlang.nif_error(:not_loaded)

  def index_resize_index(_self, _new_size), do: :erlang.nif_error(:not_loaded)

  def index_get_max_elements(_self), do: :erlang.nif_error(:not_loaded)

  def index_get_current_count(_self), do: :erlang.nif_error(:not_loaded)

  def index_get_ef_construction(_self), do: :erlang.nif_error(:not_loaded)

  def index_get_m(_self), do: :erlang.nif_error(:not_loaded)

  def float_size, do: :erlang.nif_error(:not_loaded)
end
