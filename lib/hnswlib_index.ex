defmodule HNSWLib.Index do
  @moduledoc """
  Documentation for `HNSWLib.Index`.
  """

  defstruct [:space, :dim, :reference]
  alias __MODULE__, as: T
  alias HNSWLib.Helper

  @doc """
  Construct a new Index

  ##### Positional Parameters

  - *space*: `:cosine` | `:ip` | `:l2`.

    An atom that indicates the vector space. Valid values are

      - `:cosine`, cosine space
      - `:ip`, inner product space
      - `:l2`, L2 space

  - *dim*: `non_neg_integer()`.

    Number of dimensions for each vector.

  - *max_elements*: `non_neg_integer()`.

    Number of maximum elements.

  ##### Keyword Paramters

  - *m*: `non_neg_integer()`.

    `M` is tightly connected with internal dimensionality of the data
    strongly affects the memory consumption

  - *ef_construction*: `non_neg_integer()`.

    controls index search speed/build speed tradeoff

  - *random_seed*: `non_neg_integer()`.
  - *allow_replace_deleted*: `boolean()`.
  """
  @spec new(:cosine | :ip | :l2, non_neg_integer(), non_neg_integer(), [
          {:m, non_neg_integer()},
          {:ef_construction, non_neg_integer()},
          {:random_seed, non_neg_integer()},
          {:allow_replace_deleted, boolean()}
        ]) :: {:ok, %T{}} | {:error, String.t()}
  def new(space, dim, max_elements, opts \\ [])
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 and
             is_integer(max_elements) and max_elements >= 0 do
    with {:ok, m} <- Helper.get_keyword!(opts, :m, :non_neg_integer, 16),
         {:ok, ef_construction} <-
           Helper.get_keyword!(opts, :ef_construction, :non_neg_integer, 200),
         {:ok, random_seed} <- Helper.get_keyword!(opts, :random_seed, :non_neg_integer, 100),
         {:ok, allow_replace_deleted} <-
           Helper.get_keyword!(opts, :allow_replace_deleted, :boolean, false),
         {:ok, ref} <-
           HNSWLib.Nif.index_new(
             space,
             dim,
             max_elements,
             m,
             ef_construction,
             random_seed,
             allow_replace_deleted
           ) do
      {:ok,
       %T{
         space: space,
         dim: dim,
         reference: ref
       }}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Query the index with a single vector or a list of vectors.

  ##### Positional Parameters

  - *query*: `Nx.Tensor.t() | binary() | [binary()]`.

    A vector or a list of vectors to query.

    If *query* is a list of vectors, the vectors must be of the same dimension.

  ##### Keyword Paramters

  - *k*: `pos_integer()`.

    Number of nearest neighbors to return.

  - *num_threads*: `integer()`.

    Number of threads to use.
  """
  @spec knn_query(%T{}, Nx.Tensor.t() | binary() | [binary()], [
          {:k, pos_integer()},
          {:num_threads, integer()}
          # {:filter, function()}
        ]) :: :ok | {:error, String.t()}
  def knn_query(self, query, opts \\ [])

  def knn_query(self = %T{}, query, opts) when is_binary(query) do
    with {:ok, k} <- Helper.get_keyword!(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword!(opts, :num_threads, :integer, -1),
         #  {:ok, filter} <- Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true),
         :ok <- might_be_float_data?(query),
         features = trunc(byte_size(query) / float_size()),
         {:ok, true} <- ensure_vector_dimension(self, features, true) do
      _do_knn_query(self, query, k, num_threads, nil, 1, features)
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, query, opts) when is_list(query) do
    with {:ok, k} <- Helper.get_keyword!(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword!(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true),
         {:ok, {rows, features}} <- Helper.list_of_binary(query),
         {:ok, true} <- ensure_vector_dimension(self, features, true) do
      _do_knn_query(self, IO.iodata_to_binary(query), k, num_threads, filter, rows, features)
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, query = %Nx.Tensor{}, opts) do
    with {:ok, k} <- Helper.get_keyword!(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword!(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true),
         {:ok, f32_data, rows, features} <- verify_data_tensor(self, query) do
      _do_knn_query(self, f32_data, k, num_threads, filter, rows, features)
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp _do_knn_query(self = %T{}, query, k, num_threads, filter, rows, features) do
    case HNSWLib.Nif.index_knn_query(self.reference, query, k, num_threads, filter, rows, features) do
      {:ok, labels, dists, rows, k, label_bits, dist_bits} ->
        labels = Nx.reshape(Nx.from_binary(labels, :"u#{label_bits}"), {rows, k})
        dists = Nx.reshape(Nx.from_binary(dists, :"f#{dist_bits}"), {rows, k})
        {:ok, labels, dists}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Get a list of existing IDs in the index.
  """
  @spec get_ids_list(%T{}) :: {:ok, [integer()]} | {:error, String.t()}
  def get_ids_list(self = %T{}) do
    HNSWLib.Nif.index_get_ids_list(self.reference)
  end

  @doc """
  Get the ef parameter.
  """
  @spec get_ef(%T{}) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def get_ef(self = %T{}) do
    HNSWLib.Nif.index_get_ef(self.reference)
  end

  @doc """
  Set the ef parameter.
  """
  @spec set_ef(%T{}, non_neg_integer()) :: :ok | {:error, String.t()}
  def set_ef(self = %T{}, new_ef) when is_integer(new_ef) and new_ef >= 0 do
    HNSWLib.Nif.index_set_ef(self.reference, new_ef)
  end

  @doc """
  Get the number of threads to use.
  """
  @spec get_num_threads(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_num_threads(self = %T{}) do
    HNSWLib.Nif.index_get_num_threads(self.reference)
  end

  @doc """
  Set the number of threads to use.
  """
  @spec set_num_threads(%T{}, integer()) :: {:ok, integer()} | {:error, String.t()}
  def set_num_threads(self = %T{}, new_num_threads) do
    HNSWLib.Nif.index_set_num_threads(self.reference, new_num_threads)
  end

  @doc """
  Save current index to disk.

  ##### Positional Parameters

  - *path*: `Path.t()`.

    Path to save the index to.
  """
  @spec save_index(%T{}, Path.t()) :: {:ok, integer()} | {:error, String.t()}
  def save_index(self = %T{}, path) when is_binary(path) do
    HNSWLib.Nif.index_save_index(self.reference, path)
  end

  @doc """
  Load index from disk.

  ##### Positional Parameters

  - *path*: `Path.t()`.

    Path to load the index from.

  ##### Keyword Parameters

  - *max_elements*: `non_neg_integer()`.

    Maximum number of elements to load from the index.
    If set to 0, all elements will be loaded.
    Default: 0.

  - *allow_replace_deleted*: `boolean()`.
  """
  @spec load_index(%T{}, Path.t(), [
          {:max_elements, non_neg_integer()},
          {:allow_replace_deleted, boolean()}
        ]) :: :ok | {:error, String.t()}
  def load_index(self = %T{}, path, opts \\ []) when is_binary(path) and is_list(opts) do
    with {:ok, max_elements} <- Helper.get_keyword!(opts, :max_elements, :non_neg_integer, 0),
         {:ok, allow_replace_deleted} <-
           Helper.get_keyword!(opts, :allow_replace_deleted, :boolean, false) do
      HNSWLib.Nif.index_load_index(self.reference, path, max_elements, allow_replace_deleted)
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Mark a label as deleted.

  ##### Positional Parameters

  - *label*: `non_neg_integer()`.

    Label to mark as deleted.
  """
  @spec mark_deleted(%T{}, non_neg_integer()) :: :ok | {:error, String.t()}
  def mark_deleted(self = %T{}, label) when is_integer(label) and label >= 0 do
    HNSWLib.Nif.index_mark_deleted(self.reference, label)
  end

  @doc """
  Unmark a label as deleted.

  ##### Positional Parameters

  - *label*: `non_neg_integer()`.

    Label to unmark as deleted.
  """
  @spec unmark_deleted(%T{}, non_neg_integer()) :: :ok | {:error, String.t()}
  def unmark_deleted(self = %T{}, label) when is_integer(label) and label >= 0 do
    HNSWLib.Nif.index_unmark_deleted(self.reference, label)
  end

  @doc """
  Add items to the index.

  ##### Positional Parameters

  - *data*: `Nx.Tensor.t()`.

    Data to add to the index.

  ##### Keyword Parameters

  - *ids*: `Nx.Tensor.t() | [non_neg_integer()] | nil`.

    IDs to assign to the data.

    If `nil`, IDs will be assigned sequentially starting from 0.

    Defaults to `nil`.

  - *num_threads*: `integer()`.

    Number of threads to use.

    If set to `-1`, the number of threads will be automatically determined.

    Defaults to `-1`.

  - *replace_deleted*: `boolean()`.

    Whether to replace deleted items.

    Defaults to `false`.
  """
  @spec add_items(%T{}, Nx.Tensor.t(), [
          {:ids, Nx.Tensor.t() | [non_neg_integer()] | nil},
          {:num_threads, integer()},
          {:replace_deleted, false}
        ]) :: :ok | {:error, String.t()}
  def add_items(self, data, opts \\ [])

  def add_items(self = %T{}, data = %Nx.Tensor{}, opts) when is_list(opts) do
    with {:ok, ids} <- normalize_ids(opts[:ids]),
         {:ok, num_threads} <- Helper.get_keyword!(opts, :num_threads, :integer, -1),
         {:ok, replace_deleted} <- Helper.get_keyword!(opts, :replace_deleted, :boolean, false),
         {:ok, f32_data, rows, features} <- verify_data_tensor(self, data) do
      HNSWLib.Nif.index_add_items(
        self.reference,
        f32_data,
        ids,
        num_threads,
        replace_deleted,
        rows,
        features
      )
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Retrieve items from the index using IDs.

  ##### Positional Parameters

  - *ids*: `Nx.Tensor.t() | [non_neg_integer()]`.

    IDs to retrieve.

  ##### Keyword Parameters

  - *return*: `:tensor | :list | :binary`.

    Whether to return a tensor, a list of `[numbers()]` or a list of binary.

    Defaults to `:tensor`.
  """
  @spec get_items(%T{}, Nx.Tensor.t() | [integer()], [
          {:return, :tensor | :list | :binary}
        ]) :: {:ok, [[number()]] | Nx.Tensor.t() | [binary()]} | {:error, String.t()}
  def get_items(self = %T{}, ids, opts \\ []) do
    with {:ok, ids} <- normalize_ids(ids),
         {:ok, return} <-
           Helper.get_keyword!(opts, :return, {:atom, [:tensor, :list, :binary]}, :tensor),
         {:ok, data} <- HNSWLib.Nif.index_get_items(self.reference, ids, return) do
      return_val =
        case return do
          :tensor ->
            Enum.map(data, fn bin ->
              Nx.from_binary(bin, :f32)
            end)

          :binary ->
            data

          :list ->
            data
        end

      {:ok, return_val}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Resize the index.

  ##### Positional Parameters

  - *new_size*: `non_neg_integer()`.

    New size of the index.
  """
  @spec resize_index(%T{}, non_neg_integer()) :: :ok | {:error, String.t()}
  def resize_index(self = %T{}, new_size) when is_integer(new_size) and new_size >= 0 do
    HNSWLib.Nif.index_resize_index(self.reference, new_size)
  end

  @doc """
  Get the maximum number of elements the index can hold.
  """
  @spec get_max_elements(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_max_elements(self = %T{}) do
    HNSWLib.Nif.index_get_max_elements(self.reference)
  end

  @doc """
  Get the current number of elements in the index.
  """
  @spec get_current_count(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_current_count(self = %T{}) do
    HNSWLib.Nif.index_get_current_count(self.reference)
  end

  @doc """
  Get the ef_construction parameter.
  """
  @spec get_ef_construction(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_ef_construction(self = %T{}) do
    HNSWLib.Nif.index_get_ef_construction(self.reference)
  end

  @doc """
  Get the M parameter.
  """
  @spec get_m(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_m(self = %T{}) do
    HNSWLib.Nif.index_get_m(self.reference)
  end

  defp verify_data_tensor(self = %T{}, data = %Nx.Tensor{}) do
    row_features =
      case data.shape do
        {rows, features} ->
          ensure_vector_dimension(self, features, {rows, features})

        {features} ->
          ensure_vector_dimension(self, features, {1, features})

        shape ->
          {:error,
           "Input vector data wrong shape. Number of dimensions #{tuple_size(shape)}. Data must be a 1D or 2D array."}
      end

    with {:ok, {rows, features}} <- row_features do
      {:ok, Nx.to_binary(Nx.as_type(data, :f32)), rows, features}
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  defp ensure_vector_dimension(%T{dim: dim}, dim, ret), do: {:ok, ret}

  defp ensure_vector_dimension(self = %T{}, features, _ret) do
    {:error, "Wrong dimensionality of the vectors, expect `#{self.dim}`, got `#{features}`"}
  end

  defp might_be_float_data?(data) do
    if rem(byte_size(data), float_size()) != 0 do
      {:error,
       "vector feature size should be a multiple of #{HNSWLib.Nif.float_size()} (sizeof(float))"}
    else
      :ok
    end
  end

  defp normalize_ids(ids = %Nx.Tensor{}) do
    case ids.shape do
      {_} ->
        {:ok, Nx.to_binary(Nx.as_type(ids, :u64))}

      shape ->
        {:error, "expect ids to be a 1D array, got `#{inspect(shape)}`"}
    end
  end

  defp normalize_ids(ids) when is_list(ids) do
    if Enum.all?(ids, fn x ->
         is_integer(x) and x >= 0
       end) do
      {:ok, ids}
    else
      {:error, "expect `ids` to be a list of non-negative integers"}
    end
  end

  defp normalize_ids(nil) do
    {:ok, nil}
  end

  defp float_size do
    HNSWLib.Nif.float_size()
  end
end
