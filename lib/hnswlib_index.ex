defmodule HNSWLib.Index do
  @moduledoc """
  Documentation for `HNSWLib.Index`.
  """

  defstruct [:space, :dim, :pid]
  alias __MODULE__, as: T
  alias HNSWLib.Helper

  use GenServer

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
    with {:ok, m} <- Helper.get_keyword(opts, :m, :non_neg_integer, 16),
         {:ok, ef_construction} <-
           Helper.get_keyword(opts, :ef_construction, :non_neg_integer, 200),
         {:ok, random_seed} <- Helper.get_keyword(opts, :random_seed, :non_neg_integer, 100),
         {:ok, allow_replace_deleted} <-
           Helper.get_keyword(opts, :allow_replace_deleted, :boolean, false),
         {:ok, pid} <-
           GenServer.start(
             __MODULE__,
             {space, dim, max_elements, m, ef_construction, random_seed, allow_replace_deleted}
           ) do
      {:ok,
       %T{
         space: space,
         dim: dim,
         pid: pid
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
    with {:ok, k} <- Helper.get_keyword(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         #  {:ok, filter} <- Helper.get_keyword(opts, :filter, {:function, 1}, nil, true),
         :ok <- might_be_float_data?(query),
         features = trunc(byte_size(query) / float_size()),
         {:ok, true} <- ensure_vector_dimension(self, features, true) do
      GenServer.call(self.pid, {:knn_query, query, k, num_threads, nil, 1, features})
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, query, opts) when is_list(query) do
    with {:ok, k} <- Helper.get_keyword(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword(opts, :filter, {:function, 1}, nil, true),
         {:ok, {rows, features}} <- Helper.list_of_binary(query),
         {:ok, true} <- ensure_vector_dimension(self, features, true) do
      GenServer.call(
        self.pid,
        {:knn_query, IO.iodata_to_binary(query), k, num_threads, filter, rows, features}
      )
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  def knn_query(self = %T{}, query = %Nx.Tensor{}, opts) do
    with {:ok, k} <- Helper.get_keyword(opts, :k, :pos_integer, 1),
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         {:ok, filter} <- Helper.get_keyword(opts, :filter, {:function, 1}, nil, true),
         {:ok, f32_data, rows, features} <- verify_data_tensor(self, query) do
      GenServer.call(self.pid, {:knn_query, f32_data, k, num_threads, filter, rows, features})
    else
      {:error, reason} ->
        {:error, reason}
    end
  end

  @doc """
  Get a list of existing IDs in the index.
  """
  @spec get_ids_list(%T{}) :: {:ok, [integer()]} | {:error, String.t()}
  def get_ids_list(self = %T{}) do
    GenServer.call(self.pid, :get_ids_list)
  end

  @doc """
  Get the ef parameter.
  """
  @spec get_ef(%T{}) :: {:ok, non_neg_integer()} | {:error, String.t()}
  def get_ef(self = %T{}) do
    GenServer.call(self.pid, :get_ef)
  end

  @doc """
  Set the ef parameter.
  """
  @spec set_ef(%T{}, non_neg_integer()) :: :ok | {:error, String.t()}
  def set_ef(self = %T{}, new_ef) when is_integer(new_ef) and new_ef >= 0 do
    GenServer.call(self.pid, {:set_ef, new_ef})
  end

  @doc """
  Get the number of threads to use.
  """
  @spec get_num_threads(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_num_threads(self = %T{}) do
    GenServer.call(self.pid, :get_num_threads)
  end

  @doc """
  Set the number of threads to use.
  """
  @spec set_num_threads(%T{}, integer()) :: {:ok, integer()} | {:error, String.t()}
  def set_num_threads(self = %T{}, new_num_threads) do
    GenServer.call(self.pid, {:set_num_threads, new_num_threads})
  end

  @doc """
  Save current index to disk.

  ##### Positional Parameters

  - *path*: `Path.t()`.

    Path to save the index to.
  """
  @spec save_index(%T{}, Path.t()) :: {:ok, integer()} | {:error, String.t()}
  def save_index(self = %T{}, path) when is_binary(path) do
    GenServer.call(self.pid, {:save_index, path})
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
    with {:ok, max_elements} <- Helper.get_keyword(opts, :max_elements, :non_neg_integer, 0),
         {:ok, allow_replace_deleted} <-
           Helper.get_keyword(opts, :allow_replace_deleted, :boolean, false) do
      GenServer.call(self.pid, {:load_index, path, max_elements, allow_replace_deleted})
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
    GenServer.call(self.pid, {:mark_deleted, label})
  end

  @doc """
  Unmark a label as deleted.

  ##### Positional Parameters

  - *label*: `non_neg_integer()`.

    Label to unmark as deleted.
  """
  @spec unmark_deleted(%T{}, non_neg_integer()) :: :ok | {:error, String.t()}
  def unmark_deleted(self = %T{}, label) when is_integer(label) and label >= 0 do
    GenServer.call(self.pid, {:unmark_deleted, label})
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
         {:ok, num_threads} <- Helper.get_keyword(opts, :num_threads, :integer, -1),
         {:ok, replace_deleted} <- Helper.get_keyword(opts, :replace_deleted, :boolean, false),
         {:ok, f32_data, rows, features} <- verify_data_tensor(self, data) do
      GenServer.call(
        self.pid,
        {:add_items, f32_data, ids, num_threads, replace_deleted, rows, features}
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
           Helper.get_keyword(opts, :return, {:atom, [:tensor, :list, :binary]}, :tensor) do
      GenServer.call(self.pid, {:get_items, ids, return})
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
    GenServer.call(self.pid, {:resize_index, new_size})
  end

  @doc """
  Get the maximum number of elements the index can hold.
  """
  @spec get_max_elements(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_max_elements(self = %T{}) do
    GenServer.call(self.pid, :get_max_elements)
  end

  @doc """
  Get the current number of elements in the index.
  """
  @spec get_current_count(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_current_count(self = %T{}) do
    GenServer.call(self.pid, :get_current_count)
  end

  @doc """
  Get the ef_construction parameter.
  """
  @spec get_ef_construction(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_ef_construction(self = %T{}) do
    GenServer.call(self.pid, :get_ef_construction)
  end

  @doc """
  Get the M parameter.
  """
  @spec get_m(%T{}) :: {:ok, integer()} | {:error, String.t()}
  def get_m(self = %T{}) do
    GenServer.call(self.pid, :get_m)
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

  # GenServer callbacks

  @impl true
  def init({space, dim, max_elements, m, ef_construction, random_seed, allow_replace_deleted}) do
    case HNSWLib.Nif.index_new(
           space,
           dim,
           max_elements,
           m,
           ef_construction,
           random_seed,
           allow_replace_deleted
         ) do
      {:ok, ref} ->
        {:ok, ref}

      {:error, reason} ->
        {:stop, {:error, reason}}
    end
  end

  @impl true
  def handle_call({:knn_query, data, k, num_threads, filter, rows, features}, _from, self) do
    case HNSWLib.Nif.index_knn_query(self, data, k, num_threads, filter, rows, features) do
      {:ok, labels, dists, rows, k, label_bits, dist_bits} ->
        labels = Nx.reshape(Nx.from_binary(labels, :"u#{label_bits}"), {rows, k})
        dists = Nx.reshape(Nx.from_binary(dists, :"f#{dist_bits}"), {rows, k})
        {:reply, {:ok, labels, dists}, self}

      {:error, reason} ->
        {:error, reason}
    end
  end

  @impl true
  def handle_call(
        {:add_items, f32_data, ids, num_threads, replace_deleted, rows, features},
        _from,
        self
      ) do
    {:reply,
     HNSWLib.Nif.index_add_items(
       self,
       f32_data,
       ids,
       num_threads,
       replace_deleted,
       rows,
       features
     ), self}
  end

  @impl true
  def handle_call({:get_items, ids, return}, _from, self) do
    with {:ok, data} <- HNSWLib.Nif.index_get_items(self, ids, return) do
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

      {:reply, {:ok, return_val}, self}
    else
      {:error, reason} ->
        {:reply, {:error, reason}, self}
    end
  end

  @impl true
  def handle_call(:get_ids_list, _from, self) do
    {:reply, HNSWLib.Nif.index_get_ids_list(self), self}
  end

  @impl true
  def handle_call(:get_ef, _from, self) do
    {:reply, HNSWLib.Nif.index_get_ef(self), self}
  end

  @impl true
  def handle_call({:set_ef, new_ef}, _from, self) do
    {:reply, HNSWLib.Nif.index_set_ef(self, new_ef), self}
  end

  @impl true
  def handle_call(:get_num_threads, _from, self) do
    {:reply, HNSWLib.Nif.index_get_num_threads(self), self}
  end

  @impl true
  def handle_call({:set_num_threads, new_num_threads}, _from, self) do
    {:reply, HNSWLib.Nif.index_set_num_threads(self, new_num_threads), self}
  end

  @impl true
  def handle_call({:save_index, path}, _from, self) do
    {:reply, HNSWLib.Nif.index_save_index(self, path), self}
  end

  @impl true
  def handle_call({:load_index, path, max_elements, allow_replace_deleted}, _from, self) do
    {:reply, HNSWLib.Nif.index_load_index(self, path, max_elements, allow_replace_deleted), self}
  end

  @impl true
  def handle_call({:mark_deleted, label}, _from, self) do
    {:reply, HNSWLib.Nif.index_mark_deleted(self, label), self}
  end

  @impl true
  def handle_call({:unmark_deleted, label}, _from, self) do
    {:reply, HNSWLib.Nif.index_unmark_deleted(self, label), self}
  end

  @impl true
  def handle_call({:resize_index, new_size}, _from, self) do
    {:reply, HNSWLib.Nif.index_resize_index(self, new_size), self}
  end

  @impl true
  def handle_call(:get_max_elements, _from, self) do
    {:reply, HNSWLib.Nif.index_get_max_elements(self), self}
  end

  @impl true
  def handle_call(:get_current_count, _from, self) do
    {:reply, HNSWLib.Nif.index_get_current_count(self), self}
  end

  @impl true
  def handle_call(:get_ef_construction, _from, self) do
    {:reply, HNSWLib.Nif.index_get_ef_construction(self), self}
  end

  @impl true
  def handle_call(:get_m, _from, self) do
    {:reply, HNSWLib.Nif.index_get_m(self), self}
  end
end
