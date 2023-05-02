defmodule HNSWLib.BFIndex do
  @moduledoc """
  Documentation for `HNSWLib.BFIndex`.
  """

  defstruct [:space, :dim, :pid]
  alias __MODULE__, as: T

  use GenServer

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
  @spec new(:cosine | :ip | :l2, non_neg_integer(), non_neg_integer()) :: {:ok, %T{}} | {:error, String.t()}
  def new(space, dim, max_elements)
      when (space == :l2 or space == :ip or space == :cosine) and is_integer(dim) and dim >= 0 and
             is_integer(max_elements) and max_elements >= 0 do
    with {:ok, pid} <- GenServer.start(__MODULE__, {space, dim, max_elements}) do
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

  # GenServer callbacks

  @impl true
  def init({space, dim, max_elements}) do
    case HNSWLib.Nif.bfindex_new(space, dim, max_elements) do
      {:ok, ref} ->
        {:ok, ref}

      {:error, reason} ->
        {:stop, {:error, reason}}
    end
  end
end
