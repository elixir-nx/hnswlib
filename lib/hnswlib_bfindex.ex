defmodule HNSWLib.BFIndex do
  @moduledoc """
  Documentation for `HNSWLib.BFIndex`.
  """

  defstruct [:space, :dim, :reference]
  alias __MODULE__, as: T
  alias HNSWLib.Helper

  @doc """
  Construct a new BFIndex

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
  """
  @spec new(:cosine | :ip | :l2, non_neg_integer(), non_neg_integer()) ::
          {:ok, %T{}} | {:error, String.t()}
  def new(space, dim, max_elements)
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 and
             is_integer(max_elements) and max_elements >= 0 do
    with {:ok, ref} <- HNSWLib.Nif.bfindex_new(space, dim, max_elements) do
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
  """
  @spec knn_query(%T{}, Nx.Tensor.t() | binary() | [binary()], [
          {:k, pos_integer()}
        ]) :: {:ok, Nx.Tensor.t(), Nx.Tensor.t()} | {:error, String.t()}
  def knn_query(self, query, opts \\ [])

  def knn_query(self = %T{}, query, opts) when is_binary(query) do
    k = Helper.get_keyword!(opts, :k, :pos_integer, 1)
    Helper.might_be_float_data!(query)
    features = trunc(byte_size(query) / HNSWLib.Nif.float_size())
    Helper.ensure_vector_dimension!(self, features, true)

    _do_knn_query(self, query, k, nil, 1, features)
  end

  def knn_query(self = %T{}, query, opts) when is_list(query) do
    k = Helper.get_keyword!(opts, :k, :pos_integer, 1)
    filter = Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true)
    {rows, features} = Helper.list_of_binary(query)
    Helper.ensure_vector_dimension!(self, features, true)

    _do_knn_query(self, IO.iodata_to_binary(query), k, filter, rows, features)
  end

  def knn_query(self = %T{}, query = %Nx.Tensor{}, opts) do
    k = Helper.get_keyword!(opts, :k, :pos_integer, 1)
    filter = Helper.get_keyword!(opts, :filter, {:function, 1}, nil, true)
    {f32_data, rows, features} = Helper.verify_data_tensor!(self, query)

    _do_knn_query(self, f32_data, k, filter, rows, features)
  end

  defp _do_knn_query(self, query, k, filter, rows, features) do
    case HNSWLib.Nif.bfindex_knn_query(self.reference, query, k, filter, rows, features) do
      {:ok, labels, dists, rows, k, label_bits, dist_bits} ->
        labels = Nx.reshape(Nx.from_binary(labels, :"u#{label_bits}"), {rows, k})
        dists = Nx.reshape(Nx.from_binary(dists, :"f#{dist_bits}"), {rows, k})
        {:ok, labels, dists}

      {:error, reason} ->
        {:error, reason}
    end
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

  """
  @spec add_items(%T{}, Nx.Tensor.t(), [
          {:ids, Nx.Tensor.t() | [non_neg_integer()] | nil}
        ]) :: :ok | {:error, String.t()}
  def add_items(self, data, opts \\ [])

  def add_items(self = %T{}, data = %Nx.Tensor{}, opts) when is_list(opts) do
    ids = Helper.normalize_ids!(opts[:ids])
    {f32_data, rows, features} = Helper.verify_data_tensor!(self, data)

    HNSWLib.Nif.bfindex_add_items(self.reference, f32_data, ids, rows, features)
  end

  @doc """
  Delete vectors with the given labels from the index.
  """
  def delete_vector(self = %T{}, label) when is_integer(label) do
    HNSWLib.Nif.bfindex_delete_vector(self.reference, label)
  end

  @doc """
  Get the current number of threads to use in the index.
  """
  @spec set_num_threads(%T{}, pos_integer()) :: :ok | {:error, String.t()}
  def set_num_threads(self = %T{}, num_threads)
      when is_integer(num_threads) and num_threads > 0 do
    HNSWLib.Nif.bfindex_set_num_threads(self.reference, num_threads)
  end

  @doc """
  Save current index to disk.

  ##### Positional Parameters

  - *path*: `Path.t()`.

    Path to save the index to.
  """
  @spec save_index(%T{}, Path.t()) :: :ok | {:error, String.t()}
  def save_index(self = %T{}, path) when is_binary(path) do
    HNSWLib.Nif.bfindex_save_index(self.reference, path)
  end

  @doc """
  Load index from disk.

  ##### Positional Parameters

  - *space*: `:cosine` | `:ip` | `:l2`.

    An atom that indicates the vector space. Valid values are

      - `:cosine`, cosine space
      - `:ip`, inner product space
      - `:l2`, L2 space

  - *dim*: `non_neg_integer()`.

    Number of dimensions for each vector.

  - *path*: `Path.t()`.

    Path to load the index from.

  ##### Keyword Parameters

  - *max_elements*: `non_neg_integer()`.

    Maximum number of elements to load from the index.

    If set to 0, all elements will be loaded.

    Defaults to 0.
  """
  @spec load_index(:cosine | :ip | :l2, non_neg_integer(), Path.t(), [
          {:max_elements, non_neg_integer()}
        ]) :: {:ok, %T{}} | {:error, String.t()}
  def load_index(space, dim, path, opts \\ [])
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 and
             is_binary(path) and is_list(opts) do
    max_elements = Helper.get_keyword!(opts, :max_elements, :non_neg_integer, 0)

    with {:ok, ref} <- HNSWLib.Nif.bfindex_load_index(space, dim, path, max_elements) do
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
  Get the maximum number of elements the index can hold.
  """
  @spec get_max_elements(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_max_elements(self = %T{}) do
    HNSWLib.Nif.bfindex_get_max_elements(self.reference)
  end

  @doc """
  Get the current number of elements in the index.
  """
  @spec get_current_count(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_current_count(self = %T{}) do
    HNSWLib.Nif.bfindex_get_current_count(self.reference)
  end

  @doc """
  Get the current number of threads to use in the index.
  """
  @spec get_num_threads(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_num_threads(self = %T{}) do
    HNSWLib.Nif.bfindex_get_num_threads(self.reference)
  end
end
